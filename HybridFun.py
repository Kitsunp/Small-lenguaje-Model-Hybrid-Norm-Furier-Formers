import math
import os
import random
from typing import Any, Dict, List, Optional

import datasets
from datasets import concatenate_datasets, load_dataset
from datasets import Dataset as HFDataset, IterableDataset, load_dataset
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import RMSNorm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
import multiprocessing

# CONFIGURACIÓN DE TORCH
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.suppress_errors = True

############################################
# FUNCIONES DE INICIALIZACIÓN DE CAPAS
############################################
def init_linear(layer: nn.Linear, random_factor: float = 0.02):
    gain = nn.init.calculate_gain('linear') * (1.0 + random.uniform(-random_factor, random_factor))
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

def init_embedding(embedding: nn.Embedding):
    nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

def init_gate_parameter(gate: torch.Tensor, a: float = -0.02, b: float = 0.02):
    nn.init.uniform_(gate, a=a, b=b)

############################################
# NUEVA CAPA: COLA NORMAL – CAPA LINEAL DE BAJO RANGO
############################################
class CoLA_Linear(nn.Module):
    """
    Implementación de una capa lineal según la propuesta CoLA (normal).
    Reemplaza la operación full-rank W*x por:
        h' = B(σ(Ax))
    donde A y B son matrices de bajo rango, y σ es una función de activación no lineal.
    
    Por defecto, se utiliza rank = in_features // 4.
    """
    def __init__(self, in_features: int, out_features: int, rank: Optional[int] = None, activation=F.gelu):
        super().__init__()
        if rank is None:
            rank = in_features // 4
        self.rank = rank
        self.activation = activation
        # Definición de las dos proyecciones
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=True)
        init_linear(self.A)
        init_linear(self.B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.activation(self.A(x)))

############################################
# NUEVA CAPA: COLA_FAN – CAPA LINEAL CON ANÁLISIS DE FOURIER PARA FANFORMER
############################################
class CoLA_FAN(nn.Module):
    """
    Implementación de una capa CoLA con análisis de Fourier para FANformer.
    Combina la eficiencia de CoLA con la capacidad de modelado de periodicidad de FANformer.
    
    Parámetros:
        in_features: Dimensión de entrada
        out_features: Dimensión de salida
        rank: Rango para compresión CoLA (por defecto in_features // 4)
        p: Proporción de la dimensión dedicada al modelado periódico (por defecto 0.15)
        activation: Función de activación para las proyecciones
        dropout: Tasa de dropout para regularización
        depth: Profundidad de la capa en la red (para dropout progresivo)
    """
    def __init__(self, in_features: int, out_features: int, rank: Optional[int] = None, 
                 p: float = 0.15, activation=F.gelu, dropout: float = 0.12, depth: int = 1):
        super().__init__()
        if rank is None:
            rank = in_features // 4
        self.rank = rank
        self.activation = activation
        self.p = p
        
        # Calcular dimensiones para componentes periódicos y no periódicos
        p_dim = int(out_features * p)               # Dimensión para componente periódico (antes de cos/sin)
        non_p_dim = out_features - 2 * p_dim        # Dimensión para componente no periódico
        
        # Proyecciones para componente periódico
        self.A_p = nn.Linear(in_features, rank, bias=False)
        self.B_p = nn.Linear(rank, p_dim, bias=False)  # Sin bias para transformación periódica
        
        # Proyecciones para componente no periódico (CoLA estándar)
        self.A_np = nn.Linear(in_features, rank, bias=False)
        self.B_np = nn.Linear(rank, non_p_dim, bias=True)
        
        # Dropout para regularización
        self.dropout_p = progressive_dropout(dropout, depth)
        self.dropout_np = progressive_dropout(dropout, depth)
        
        # Inicialización
        init_linear(self.A_p)
        init_linear(self.B_p)
        init_linear(self.A_np)
        init_linear(self.B_np)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Componente periódico con dropout
        p_activation = self.activation(self.A_p(x))
        p_activation = self.dropout_p(p_activation)
        p_proj = self.B_p(p_activation)
        
        # Componente no periódico con dropout
        np_activation = self.activation(self.A_np(x))
        np_activation = self.dropout_np(np_activation)
        np_proj = self.B_np(np_activation)
        
        # Combinar usando transformaciones de Fourier (cos/sin) y componente regular
        return torch.cat([torch.cos(p_proj), torch.sin(p_proj), np_proj], dim=-1)

############################################
# UTILIDAD: CREACIÓN DE DROPOUT PROGRESIVO
############################################
def progressive_dropout(p: float, depth: int) -> nn.Dropout:
    if p == 0.0:
        return nn.Dropout(0.0)
    return nn.Dropout(p * (1 + depth * 0.04))

############################################
# UTILIDADES: ROPE UNIFICADO CON PRECÁLCULO
############################################
def get_rope_buffer(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float().unsqueeze(1)
    sinusoid_inp = pos * inv_freq.unsqueeze(0)
    cos = torch.cos(sinusoid_inp).to(dtype)
    sin = torch.sin(sinusoid_inp).to(dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return cos, sin

def apply_rope_vectorized(x: torch.Tensor) -> torch.Tensor:
    B, num_heads, T, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim debe ser par para RoPE")
    cos, sin = get_rope_buffer(T, head_dim, x.device, x.dtype)
    x_reshaped = x.view(B, num_heads, T, head_dim // 2, 2)
    x_even = x_reshaped[..., 0]
    x_odd = x_reshaped[..., 1]
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    result = x_rotated.flatten(-2)
    return result

############################################
# GATED RESIDUALS
############################################
class GatedResidual(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.12, depth: int = 1, init_value: float = 0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(1, 1, d_model) * init_value)
        self.dropout = progressive_dropout(dropout, depth)
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Aplicar dropout antes + sigmoid para el gate
        residual_dropped = self.dropout(residual)
        gate_value = torch.sigmoid(self.gate)
        gated_residual = gate_value * residual_dropped
        return x + gated_residual

############################################
# MÓDULO AUXILIAR: GQA FAN LINEAR
############################################
class GQAFANLinear(nn.Module):
    """
    Proyección de GQA utilizando CoLA_FAN para FANformer.
    Divide la proyección en grupos, usando internamente una capa CoLA_FAN.
    
    Se espera que out_features sea divisible por num_heads.
    
    Parámetros:
        in_features: Dimensión de entrada
        out_features: Dimensión de salida 
        num_heads: Número de cabezales de atención
        num_gqa_groups: Número de grupos para GQA
        p: Proporción de la dimensión dedicada al modelado periódico
        divide_dim: Si se debe dividir la dimensión (por defecto False)
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int, 
                 num_gqa_groups: int, p: float = 0.15, divide_dim: bool = False):
        super().__init__()
        if out_features % num_heads != 0:
            raise ValueError("out_features debe ser divisible por num_heads")
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.rep_factor = num_heads // num_gqa_groups

        self.divide_factor = 1
        self.head_dim = (out_features // num_heads) // self.divide_factor

        self.inter_dim = num_gqa_groups * self.head_dim
        # Usamos CoLA_FAN en lugar de CoLA_Linear:
        self.linear = CoLA_FAN(in_features, self.inter_dim, rank=in_features // 4, p=p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        out = self.linear(x)
        out = out.view(B, T, self.num_gqa_groups, self.head_dim)
        out = out.repeat(1, 1, self.rep_factor, 1)
        out = out.view(B, T, self.num_heads, self.head_dim)
        return out

############################################
# MÓDULO: ATENCIÓN MULTI-CABEZA CON FANFORMER
############################################
class FANformerMultiheadAttention(nn.Module):
    """
    Implementación de la atención multi-cabeza con FANformer.
    Aplica normalización a Q, K, V individualmente y utiliza unpadding para mejorar el rendimiento.
    Incorpora modelado de periodicidad a través de proyecciones CoLA_FAN.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.12, use_rope: bool = True,
                 layer_index: int = 1, max_seq_len: int = 512, p: float = 0.15,
                 num_gqa_groups: Optional[int] = None, debug: bool = True,
                 use_pre_norm: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.debug = debug
        self.layer_name = f"Layer_{layer_index}"
        self.layer_index = layer_index
        self.use_pre_norm = use_pre_norm
        self.p = p  # Proporción para periodicidad
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible por num_heads")
        
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        
        if num_gqa_groups is None:
            num_gqa_groups = num_heads

        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            self.flash_attn_func = flash_attn_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(f"Error al inicializar FlashAttention: {e}")

        # Para el unpadding
        try:
            from flash_attn.bert_padding import unpad_input, pad_input
            self.unpad_input = unpad_input
            self.pad_input = pad_input
        except ImportError as e:
            raise ImportError(f"Error al importar funciones de padding: {e}")

        # Inicialización de parámetros de escala
        self.ssmax_scale = nn.Parameter(torch.ones(num_heads, dtype=torch.bfloat16) * 0.168)
        nn.init.uniform_(self.ssmax_scale, a=0.166, b=0.170)
        self.register_buffer('seq_scale', torch.log(torch.tensor(max_seq_len, dtype=torch.bfloat16)))
        
        # Capas de normalización para la entrada (Pre-Norm en primer bloque o QKV-Norm para los demás)
        self.norm = nn.RMSNorm(embed_dim, eps=1e-8)
        
        # Capas de dropout
        self.attention_dropout = progressive_dropout(dropout, depth=1)
        self.projection_dropout = progressive_dropout(dropout * 1.1, depth=1)
        self.output_dropout = progressive_dropout(dropout, depth=1)

        # Proyecciones para Q, K, V usando GQAFANLinear (implementación FANformer)
        self.Wq = GQAFANLinear(embed_dim, embed_dim, num_heads, num_gqa_groups, p=p)
        self.Wk = GQAFANLinear(embed_dim, embed_dim, num_heads, num_gqa_groups, p=p)
        self.Wv = GQAFANLinear(embed_dim, embed_dim, num_heads, num_gqa_groups, p=p)

        # Proyección de salida (se mantiene como CoLA_Linear)
        self.out_proj = CoLA_Linear(embed_dim, embed_dim, rank=embed_dim // 4)

    def scaled_dot_product_attention_flash_unpadded(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                                   attention_mask: Optional[torch.Tensor] = None,
                                                   is_causal: bool = False) -> torch.Tensor:
        B, H, S, D = q.shape  # batch, heads, sequence length, head dimension
        
        if attention_mask is None:
            # Si no hay máscara de atención, usamos la versión regular
            return self.scaled_dot_product_attention_flash(q, k, v, mask=None, is_causal=is_causal)
        
        # Convertir las tensiones a [B, S, H, D] para unpad_input
        q_unpad = q.permute(0, 2, 1, 3)  # [B, S, H, D]
        k_unpad = k.permute(0, 2, 1, 3)  # [B, S, H, D]
        v_unpad = v.permute(0, 2, 1, 3)  # [B, S, H, D]
        
        # Preparar máscara: convertir a bool si es necesario
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # Hacer unpadding de los tensores
        q_unpadded, indices_q, cu_seqlens_q, max_seqlen_q, _ = self.unpad_input(q_unpad, attention_mask)
        k_unpadded, indices_k, cu_seqlens_k, max_seqlen_k, _ = self.unpad_input(k_unpad, attention_mask)
        v_unpadded, _, _, _, _ = self.unpad_input(v_unpad, attention_mask)
        
        # Reacomodar para flash_attn_varlen_func: [Total, H, D]
        q_unpadded = q_unpadded.reshape(-1, H, D)
        k_unpadded = k_unpadded.reshape(-1, H, D)
        v_unpadded = v_unpadded.reshape(-1, H, D)
        
        # Normalizar vectores Q y K para mejorar estabilidad numérica
        q_norm = F.normalize(q_unpadded, p=2, dim=-1)
        k_norm = F.normalize(k_unpadded, p=2, dim=-1)
        
        # Ajustar q con factor de escala
        s = self.ssmax_scale.view(1, H, 1)
        q_adjusted = q_norm * (self.seq_scale * s)
        
        # Factor de escala para softmax
        softmax_scale = 1.0 / math.sqrt(D)
        
        try:
            # Usar flash attention sin padding
            output_unpadded = self.flash_attn_varlen_func(
                q_adjusted, k_norm, v_unpadded,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=self.attention_dropout.p,
                softmax_scale=softmax_scale,
                causal=is_causal
            )
            
            # Volver a aplicar padding
            output_padded = self.pad_input(output_unpadded, indices_q, B, S)
            
            # Reorganizar a [B, H, S, D]
            output = output_padded.reshape(B, S, H, D).permute(0, 2, 1, 3)
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"Error en flash_attn_varlen_func: {e}")

    def scaled_dot_product_attention_flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                           mask: Optional[torch.Tensor] = None,
                                           is_causal: bool = False) -> torch.Tensor:
        # Normalizar vectores Q y K para mejorar estabilidad numérica
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # Ajustar q con factor de escala
        s = self.ssmax_scale.view(-1, 1, 1)
        q_adjusted = q_norm * (self.seq_scale * s)
        
        # Preparar tensores para Flash Attention (requiere shape [B, S, H, D])
        q_trans = q_adjusted.permute(0, 2, 1, 3)
        k_trans = k_norm.permute(0, 2, 1, 3)
        v_trans = v.permute(0, 2, 1, 3)
        
        # Verificar dimensiones
        if q_trans.size(-1) != k_trans.size(-1):
            raise ValueError(f"Las dimensiones de head no coinciden: q={q_trans.size(-1)}, k={k_trans.size(-1)}")
        
        # Factor de escala para softmax
        softmax_scale = 1.0 / math.sqrt(q_trans.size(-1))
        
        try:
            # Aplicar Flash Attention
            output = self.flash_attn_func(
                q_trans, k_trans, v_trans,
                dropout_p=self.attention_dropout.p,
                softmax_scale=softmax_scale,
                causal=is_causal
            )
            
            if output is None:
                raise ValueError("flash_attn_func devolvió None. Verifica las dimensiones y tipos de los tensores de entrada.")
            
            # Volver a la forma original y aplicar dropout
            output = output.permute(0, 2, 1, 3)
            return output
            
        except Exception as e:
            raise RuntimeError(f"Error en flash_attn_func: {e}")

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, causal: bool = True) -> torch.Tensor:
        B, T, _ = X.shape
        
        # Implementación de HybridNorm*
        if self.use_pre_norm:
            # Primer bloque: Pre-Norm en atención
            X_norm = self.norm(X)
            # Proyecciones para Q, K, V con FANformer
            Q = self.Wq(X_norm)  # [B, T, num_heads, head_dim]
            K = self.Wk(X_norm)  # [B, T, num_heads, head_dim]
            V = self.Wv(X_norm)  # [B, T, num_heads, head_dim]
        else:
            # Otros bloques: QKV-Norm
            Q = self.Wq(self.norm(X))  # [B, T, num_heads, head_dim]
            K = self.Wk(self.norm(X))  # [B, T, num_heads, head_dim]
            V = self.Wv(self.norm(X))  # [B, T, num_heads, head_dim]
        
        # Permutar a formato [B, num_heads, T, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # Aplicar RoPE si está activado
        if self.use_rope:
            Q = apply_rope_vectorized(Q)
            K = apply_rope_vectorized(K)
        
        # Convertir a bfloat16 para flash attention
        Q = Q.to(torch.bfloat16)
        K = K.to(torch.bfloat16)
        V = V.to(torch.bfloat16)
        
        # Procesar la secuencia utilizando unpadding si hay máscara de atención
        if attention_mask is not None:
            attn_output = self.scaled_dot_product_attention_flash_unpadded(
                Q, K, V, 
                attention_mask=attention_mask, 
                is_causal=causal
            )
        else:
            # Si no hay máscara, usar la versión regular
            attn_output = self.scaled_dot_product_attention_flash(
                Q, K, V, 
                mask=None, 
                is_causal=causal
            )
        
        # Aplicar dropout
        attn_output = self.attention_dropout(attn_output)
        
        # Reorganizar la salida y aplicar proyección final
        out = attn_output.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, T, self.embed_dim)
        out = self.output_dropout(self.out_proj(out))
        
        return out

############################################
# BLOQUE DEL FANFORMER: CAPA CON ATENCIÓN Y MLP (Decoder-Only)
############################################
class FANformerLayer(nn.Module):
    """
    Implementación de capa de transformador con FANformer.
    Similar a RegularTransformerLayer pero utiliza FANformerMultiheadAttention.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.12,
                 layer_index: int = 1, num_gqa_groups: Optional[int] = None, 
                 is_first_layer: bool = False, p: float = 0.15):
        super().__init__()
        self.is_first_layer = is_first_layer
        
        # En HybridNorm*, el primer bloque usa Pre-Norm en MHA
        # Usamos FANformerMultiheadAttention en lugar de RegularMultiheadAttention
        self.attn = FANformerMultiheadAttention(
            embed_dim, num_heads, dropout=dropout, use_rope=True,
            layer_index=layer_index, num_gqa_groups=num_gqa_groups,
            use_pre_norm=is_first_layer, p=p
        )
        
        # Residual connection para atención
        self.gated_residual_attn = GatedResidual(embed_dim, dropout=dropout, depth=1)
        
        # Post-Norm para FFN (HybridNorm)
        self.ffn_norm = nn.RMSNorm(embed_dim, eps=1e-8)
        self.mlp = SwiGLU(embed_dim, ff_dim, dropout, depth=1)
        
        # Residual connection para FFN
        self.gated_residual_mlp = GatedResidual(embed_dim, dropout=dropout, depth=1)
        
        # Post-Norm final (HybridNorm)
        self.post_ffn_norm = nn.RMSNorm(embed_dim, eps=1e-8)

    def forward(self, x: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Bloque de atención con HybridNorm
        if self.is_first_layer:
            # Primer bloque: Pre-Norm + QKV-Norm
            attn_out = self.attn(x, tgt_mask)
            x = self.gated_residual_attn(x, attn_out)
        else:
            # Otros bloques: QKV-Norm
            attn_out = self.attn(x, tgt_mask)
            x = self.gated_residual_attn(x, attn_out)
        
        # Bloque FFN con Post-Norm (HybridNorm)
        ffn_input = self.ffn_norm(x)
        ffn_out = self.mlp(ffn_input)
        x = self.gated_residual_mlp(x, ffn_out)
        x = self.post_ffn_norm(x)
        
        return x

############################################
# NUEVO MÓDULO: SWIGLU CON COLA (MLP)
############################################
class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.12, depth: int = 1):
        super().__init__()
        # Reemplazamos fc1 y fc2 por CoLA_Linear
        self.fc1 = CoLA_Linear(in_features, hidden_features * 2, rank=in_features // 4)
        self.fc2 = CoLA_Linear(hidden_features, in_features, rank=hidden_features // 4)
        self.dropout = progressive_dropout(dropout, depth)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_out = x1 * F.silu(x2)
        x_out = self.dropout(x_out)
        return self.fc2(x_out)

############################################
# FANFORMER DECODER CON RECURRENT DEPTH (Decoder-Only)
############################################
class FANformerDecoder(nn.Module):
    """
    Implementación del decoder FANformer con recurrent depth.
    """
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.12,
                 num_gqa_groups: Optional[int] = None, p: float = 0.15,
                 use_checkpoint: bool = True, skip_every: int = 3):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.skip_every = skip_every
        
        # Crear capas de FANformer con tratamiento especial para el primer bloque (HybridNorm*)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_first_layer = (i == 0)  # Identificar si es el primer bloque para HybridNorm*
            self.layers.append(
                FANformerLayer(
                    embed_dim, num_heads, ff_dim,
                    dropout=dropout * (1 + i * 0.05),
                    layer_index=i+1,
                    num_gqa_groups=num_gqa_groups,
                    is_first_layer=is_first_layer,
                    p=p
                )
            )
        
        num_skips = num_layers // skip_every
        
        # Mejora 3: Inicialización adaptativa de skip gates basada en la distancia
        self.skip_gates = nn.ParameterList([    
            nn.Parameter(torch.ones(1, 1, embed_dim) * (0.15 - i * 0.01))  # Valores decrecientes con la profundidad
            for i in range(num_skips)
        ])
        
        # Mejora 4: Inicialización mejorada para mayor estabilidad
        for i, gate in enumerate(self.skip_gates):
            # Usar inicialización normal para algunos gates y uniforme para otros
            if i % 2 == 0:
                nn.init.normal_(gate, mean=0.05, std=0.01)
            else:
                init_gate_parameter(gate, a=0.04, b=0.10)  # Rango más estrecho
        
        self.skip_norms = nn.ModuleList([
            nn.RMSNorm(embed_dim, eps=1e-8)
            for _ in range(num_skips)
        ])
        self.skip_dropouts = nn.ModuleList([
            progressive_dropout(dropout * 0.8, depth=i+1)
            for i in range(num_skips)
        ])

        self.pre_norm = nn.RMSNorm(embed_dim, eps=1e-8)
        self.dropout = progressive_dropout(dropout, depth=1)
        self.layer_norm = nn.RMSNorm(embed_dim, eps=1e-8)
        self.extra_decoder_dropout = progressive_dropout(0.15, depth=1)

    def forward(self, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        layer_states = []
        
        for i, layer in enumerate(self.layers):
            if i % self.skip_every == 0:
                layer_states.append(output)
                
            if self.use_checkpoint:
                output = checkpoint.checkpoint(lambda x: layer(x, tgt_mask=tgt_mask),
                                               output, use_reentrant=False)
            else:
                output = layer(output, tgt_mask=tgt_mask)
                
            if (i + 1) % self.skip_every == 0 and i // self.skip_every < len(self.skip_gates):  
                skip_idx = i // self.skip_every
                skip_state = layer_states[skip_idx]
                skip_state = self.skip_norms[skip_idx](skip_state)
                
                # Mejora 2: Modificar orden del dropout (primero dropout, luego gate)
                skip_state_dropped = self.skip_dropouts[skip_idx](skip_state)
                
                # Mejora 1: Aplicar sigmoid a los skip gates
                skip_gate = torch.sigmoid(self.skip_gates[skip_idx])
                
                output = output + skip_gate * skip_state_dropped
                
        output = self.pre_norm(output)
        output = self.dropout(output)
        output = self.layer_norm(output)
        output = self.extra_decoder_dropout(output)
        return output

############################################
# MODELO TEXT-ONLY (DECODER-ONLY)
############################################
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int,
                 num_heads: int, num_decoder_layers: int, ff_dim: int, dropout: float = 0.12,
                 num_gqa_groups: Optional[int] = None, p: float = 0.15):
        super().__init__()
        self.epsilon = 1e-8
        self.dropout_rate = dropout
        self.text_feat_norm = nn.RMSNorm(embed_dim, eps=self.epsilon)
        print("[DEBUG] MultiModalModel: modo decoder-only activado")
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        init_embedding(self.decoder_embedding)
        self.emb_dropout = progressive_dropout(self.dropout_rate, depth=1)
        self.decoder_input_norm = nn.RMSNorm(embed_dim, eps=self.epsilon)
        
        print(f"[DEBUG] Usando FANformer con p={p}")
        self.decoder = FANformerDecoder(
            num_decoder_layers, embed_dim, num_heads, ff_dim, 
            dropout=self.dropout_rate, num_gqa_groups=num_gqa_groups, p=p
        )
        
        self.output_proj = CoLA_Linear(embed_dim, vocab_size, rank=embed_dim // 4)
        self.tokenizer = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        dec_emb = self.decoder_embedding(tokens)
        dec_emb = self.emb_dropout(dec_emb)
        dec_emb = self.decoder_input_norm(dec_emb)
        dec_out = self.decoder(dec_emb)
        logits = self.output_proj(dec_out)
        return logits

    def sample_next_token(self, logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
        B, V = logits.shape
        beam_width = min(5, V)
        if hasattr(self, '_generated_history') and self._generated_history:
            penalty_mask = torch.ones_like(logits)
            history = self._generated_history
            max_history = min(20, len(history))
            for pos, token in enumerate(history[-max_history:]):
                distance = max_history - pos
                decay_factor = 0.8 ** (distance / 5)
                penalty = 1.0 + (0.85 * decay_factor)
                token_positions = (token == torch.arange(V, device=logits.device))
                penalty_mask[:, token_positions] *= penalty
            logits = logits / penalty_mask
        logits = logits / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, V))
            kth_value = values[:, -1].unsqueeze(1)
            logits[logits < kth_value] = -float('inf')
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        log_probs = F.log_softmax(logits, dim=-1)
        top_log_probs, top_next_tokens = torch.topk(log_probs, beam_width, dim=-1)
        if not hasattr(self, '_beam_scores') or self._beam_scores is None:
            self._beam_scores = top_log_probs[:, 0:1]
            return top_next_tokens[:, 0:1]
        current_scores = self._beam_scores
        candidate_scores = current_scores + top_log_probs
        best_scores, best_indices = torch.max(candidate_scores, dim=1, keepdim=True)
        next_token = torch.gather(top_next_tokens, 1, best_indices)
        self._beam_scores = best_scores
        return next_token

    def forward_inference(self, prompt_tokens: torch.Tensor, max_length: int = 100, min_length: int = 1,
                          top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        if self.tokenizer is None:
            raise ValueError("El tokenizador no ha sido asignado al modelo. Asigna 'model.tokenizer = tokenizer'.")
        self._generated_history = []
        self._beam_scores = None
        generated = prompt_tokens
        cur_length = generated.size(1)
        for _ in range(max_length):
            dec_emb = self.decoder_embedding(generated)
            dec_emb = self.emb_dropout(dec_emb)
            dec_emb = self.decoder_input_norm(dec_emb)
            dec_out = self.decoder(dec_emb)
            logits = self.output_proj(dec_out)
            next_token_logits = logits[:, -1, :]
            next_token = self.sample_next_token(next_token_logits, top_k=top_k, top_p=top_p, temperature=temperature)
            if next_token.dim() > 1:
                token_list = next_token.squeeze().tolist()
                if isinstance(token_list, int):
                    token_list = [token_list]
            else:
                token_list = [next_token.item()]
            self._generated_history.extend(token_list)
            generated = torch.cat([generated, next_token], dim=1)
            cur_length += 1
            if cur_length > min_length and (next_token == self.tokenizer.eos_token_id).all():
                break
        del self._generated_history
        self._beam_scores = None
        return generated

############################################
# MÓDULOS DE DATOS: SOLO TEXTO
############################################

class TextDataset(TorchDataset):
    def __init__(self, hf_dataset: HFDataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

class FinewebTextDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, max_seq_len: int, batch_size: int, max_examples: int = 10000, val_split_ratio: float = 0.05):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.max_examples = max_examples
        self.val_split_ratio = val_split_ratio
        self.train_dataset: Optional[TextDataset] = None
        self.val_dataset: Optional[TextDataset] = None
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100349
        self.dataset_info = {
            "source": "HuggingFaceFW/fineweb-edu",
            "original_license": "ODC-By v1.0",
            "terms_of_use": "CommonCrawl Terms of Use",
            "modifications": "Text cleaning applied, removed technical elements"
        }

    def clean_text(self, text: str) -> str:
        import re
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'http[s]?://(?:[a-zA-Z0-9./?=&_-]+)', '', text)
        text = re.sub(r's3://commoncrawl/crawl-data/.*\n?', '', text)
        text = re.sub(r'CC-MAIN-\d{4}-\d{2}', '', text)
        text = re.sub(r'<urn:uuid:[a-f0-9-]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def setup(self, stage: Optional[str] = None):
        import json
        from datetime import datetime
        
        streaming_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name="sample-10BT", 
            split="train", 
            streaming=True
        )
        
        if not isinstance(streaming_dataset, IterableDataset):
            raise ValueError("Expected IterableDataset")

        examples: List[Dict[str, str]] = []
        for i, example in enumerate(streaming_dataset):
            if i >= self.max_examples:
                break
            text = example.get("text", "")
            url = example.get("url", "")
            cleaned_text = self.clean_text(text)
            if cleaned_text and len(cleaned_text.split()) > 10:
                examples.append({
                    "text": cleaned_text,
                    "url": url
                })
        
        attribution_info = {
            "source": self.dataset_info["source"],
            "original_license": self.dataset_info["original_license"],
            "terms_of_use": self.dataset_info["terms_of_use"],
            "modifications": self.dataset_info["modifications"],
            "processed_examples": len(examples),
            "processing_date": str(datetime.now()),
            "min_words_filter": 10
        }
        
        with open('dataset_attribution.json', 'w') as f:
            json.dump(attribution_info, f, indent=2)
        
        examples_for_training = [{"text": ex["text"]} for ex in examples]
        full_dataset = HFDataset.from_list(examples_for_training)
        
        split_dataset = full_dataset.train_test_split(
            test_size=self.val_split_ratio, 
            seed=42
        )
        
        self.train_dataset = TextDataset(split_dataset['train'])
        self.val_dataset = TextDataset(split_dataset['test'])

    def collate_fn(self, batch: List[Dict[str, str]]) -> Optional[Dict[str, torch.Tensor]]:
        texts = [example["text"] for example in batch]
        try:
            encodings = self.tokenizer(
                texts, 
                truncation=True, 
                max_length=self.max_seq_len, 
                padding="max_length", 
                return_tensors="pt"
            )
            tokens = encodings["input_ids"]
            return {"text": tokens}
        except Exception as e:
            print(f"Error en tokenización: {e}")
            return None

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup() first.")
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

############################################
# LIGHTNING MODULE: ENTRENAMIENTO DEL MODELO EN MODO TEXTO
############################################
from transformers.trainer_pt_utils import get_parameter_names
import bitsandbytes as bnb  # Asegúrate de tener instalada una versión reciente

class MultiModalLightningModule(pl.LightningModule):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int,
                 num_heads: int, num_decoder_layers: int, ff_dim: int,
                 lr: float = 1e-4, dropout: float = 0.12,
                 NUM_GQA_GROUPS: int = 4, p: float = 0.15):
        super().__init__()
        self.save_hyperparameters()
        try:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
            self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100349
        except Exception as e:
            print(f"[WARNING] Error al cargar tokenizer: {e}")
            self.pad_token_id = 100349
            tokenizer = None
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        self.model = MultiModalModel(
            vocab_size, embed_dim, max_seq_len,
            num_heads, num_decoder_layers, ff_dim,
            dropout=dropout, num_gqa_groups=NUM_GQA_GROUPS, p=p
        )
        self.model.tokenizer = tokenizer
        self.model = torch.compile(self.model, backend="inductor",
                                   mode="max-autotune", fullgraph=True)
        self.lr = lr
        self.next_token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.12)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        text = batch["text"].to(self.device)
        logits = self.model(text)
        next_token_logits = logits[:, :-1]
        next_token_targets = text[:, 1:]
        loss = self.next_token_loss_fn(
            next_token_logits.reshape(-1, self.vocab_size),
            next_token_targets.reshape(-1)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        text = batch["text"].to(self.device)
        logits = self.model(text)
        next_token_logits = logits[:, :-1]
        next_token_targets = text[:, 1:]
        loss = self.next_token_loss_fn(
            next_token_logits.reshape(-1, self.vocab_size),
            next_token_targets.reshape(-1)
        )
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        model = self.model
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [n for n in decay_parameters if "bias" not in n]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": 0.04,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.002,
            },
        ]

        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-8
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=67000,
            T_mult=2,
            eta_min=self.lr / 5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation_loss",
                "interval": "step",
                "frequency": 1
            }
        }

############################################
# BLOQUE PRINCIPAL: TRAINING E INFERENCIA
############################################
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Entrenamiento o Inferencia del modelo FANformer (preentrenamiento)"
    )
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--epochs_text", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=100, help="Longitud máxima a generar (tokens adicionales)")
    parser.add_argument("--min_length", type=int, default=1, help="Longitud mínima a generar antes de considerar EOS")
    parser.add_argument("--top_k", type=int, default=100, help="Valor de top_k para sampling")
    parser.add_argument("--top_p", type=float, default=0.85, help="Valor de top_p para sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Factor de temperatura para escalado de logits")
    parser.add_argument("--p", type=float, default=0.15, help="Proporción de modelado periódico para FANformer")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    
    config = {
        "VOCAB_SIZE": tokenizer.vocab_size,
        "EMBED_DIM": 768,
        "NUM_HEADS": 12,
        "NUM_DECODER_LAYERS": 12,
        "FF_DIM": 2048, 
        "LR": 3e-4,
        "batch_size": args.batch_size,
        "NUM_GQA_GROUPS": 6,
        "BASE_DROPOUT": 0.1,
        "p": args.p
    }
    mode_config = {
        "MAX_SEQ_LEN": 1024,
        "max_examples": 1100000
    }

    if args.mode == "train":
        print(f"[DEBUG] Entrenamiento en modo DECODER-ONLY utilizando FineWeb-Edu (sample-10BT)")
        print(f"[DEBUG] Utilizando FANformer con p={args.p}")
            
        data_module = FinewebTextDataModule(tokenizer, max_seq_len=mode_config["MAX_SEQ_LEN"],
                                              batch_size=config["batch_size"],
                                              max_examples=mode_config["max_examples"])
        model_module = MultiModalLightningModule(vocab_size=config["VOCAB_SIZE"],
                                                 embed_dim=config["EMBED_DIM"],
                                                 max_seq_len=mode_config["MAX_SEQ_LEN"],
                                                 num_heads=config["NUM_HEADS"],
                                                 num_decoder_layers=config["NUM_DECODER_LAYERS"],
                                                 ff_dim=config["FF_DIM"],
                                                 lr=config["LR"],
                                                 dropout=config["BASE_DROPOUT"],
                                                 NUM_GQA_GROUPS=config["NUM_GQA_GROUPS"],
                                                 p=config["p"])
        trainer_config = {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "precision": "bf16-true",
            "accumulate_grad_batches": 1,
            "enable_checkpointing": False,
            "check_val_every_n_epoch": 1,
            "gradient_clip_val": 1
            #"profiler": "simple"
        }
        trainer = pl.Trainer(max_epochs=args.epochs_text, **trainer_config)
        trainer.fit(model_module, datamodule=data_module)
        os.makedirs('./modelo_final', exist_ok=True)
        final_ckpt = f"./modelo_final/modelo_final_fanformer.ckpt"
        trainer.save_checkpoint(final_ckpt)
        print(f"[DEBUG] Modelo guardado exitosamente en {final_ckpt}")
    else:
        # Inferencia
        checkpoint_path = f"./modelo_final/modelo_final_fanformer.ckpt"
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] No se encontró el checkpoint en {checkpoint_path}")
            checkpoint_path = "./modelo_final/modelo_final.ckpt"
            print(f"[INFO] Intentando con modelo genérico: {checkpoint_path}")
            
        print(f"[DEBUG] Modo de inferencia activado utilizando FANformer con p={args.p}")
        model_inference = MultiModalLightningModule(
            vocab_size=config["VOCAB_SIZE"],
            embed_dim=config["EMBED_DIM"],
            max_seq_len=mode_config["MAX_SEQ_LEN"],
            num_heads=config["NUM_HEADS"],
            num_decoder_layers=config["NUM_DECODER_LAYERS"],
            ff_dim=config["FF_DIM"],
            lr=config["LR"],
            dropout=config["BASE_DROPOUT"],
            NUM_GQA_GROUPS=config["NUM_GQA_GROUPS"],
            p=config["p"]
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_inference.load_state_dict(ckpt["state_dict"])
        model_inference.eval()
        model_inference.to(torch.bfloat16)
        model_inference.to("cuda" if torch.cuda.is_available() else "cpu")
        while True:
            prompt = input("\nIngresa tu prompt (o 'salir' para terminar): ")
            if prompt.lower() == "salir":
                break
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                generated_ids = model_inference.model.forward_inference(
                    input_ids,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature
                )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("\n=== Resultado ===")
            print(f"Prompt: {prompt}")
            print(f"Texto generado: {generated_text}")
            print("\n" + "="*50)
        print("\n¡Gracias por usar el modelo de inferencia!")