# Imports de la biblioteca estándar
import math
import os
import random
import json
import tarfile
import shutil
import io
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

# Imports de terceros
import numpy as np
import datasets
from datasets import concatenate_datasets, load_dataset
from datasets import Dataset as HFDataset, IterableDataset
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import RMSNorm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import webdataset as wds
from sklearn.model_selection import StratifiedShuffleSplit
# Configuración de entorno y de Torch
os.environ["HF_TOKEN"] = "Tu token"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
torch.set_default_dtype(torch.float16)
############################################
# IMPLEMENTACIÓN DEL OPTIMIZADOR MUON
############################################
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Iteración Newton-Schulz para calcular la potencia cero / ortogonalización de G.
    """
    assert len(G.shape) == 2
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = (b * A + c * A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    
    return X

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Usar el mismo enfoque que arriba: verificar si muon_params no es None
        if muon_params is not None:
            for p in muon_params:
                assert p.ndim == 2, f"Expected 2D parameter for Muon, got {p.ndim}D"
                self.state[p]["use_muon"] = True
        
        # Verificar si adamw_params no es None
        if adamw_params is not None:
            for p in adamw_params:
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        """
        Ajusta la tasa de aprendizaje.
        """
        A, B = param_shape[:2]
        adjusted_factor = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_factor
        return adjusted_lr

    def step(self, closure=None):
        """Realizar un paso de optimización."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
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
    
    Esta implementación omite el dropout interno ya que la regularización ya se aplica en las
    capas superiores (FANformerMultiheadAttention y flash attention). Esto evita una
    regularización excesiva que podría limitar la capacidad de aprendizaje del modelo.
    
    Parámetros:
        in_features: Dimensión de entrada
        out_features: Dimensión de salida
        rank: Rango para compresión CoLA (por defecto in_features // 4)
        p: Proporción de la dimensión dedicada al modelado periódico (por defecto 0.15)
        activation: Función de activación para las proyecciones
        depth: Profundidad de la capa en la red (mantenido para compatibilidad)
    """
    def __init__(self, in_features: int, out_features: int, rank: Optional[int] = None, 
                 p: float = 0.15, activation=F.gelu, dropout: float = 0.0, depth: int = 1):
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
        
        # Se elimina el dropout interno para evitar regularización excesiva
        # ya que el dropout se aplica en capas superiores (FANformerMultiheadAttention)
        
        # Inicialización
        init_linear(self.A_p)
        init_linear(self.B_p)
        init_linear(self.A_np)  
        init_linear(self.B_np)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Componente periódico sin dropout
        p_activation = self.activation(self.A_p(x))
        p_proj = self.B_p(p_activation)
        
        # Componente no periódico sin dropout
        np_activation = self.activation(self.A_np(x))
        np_proj = self.B_np(np_activation)
        
        # Combinar usando transformaciones de Fourier (cos/sin) y componente regular
        return torch.cat([torch.cos(p_proj), torch.sin(p_proj), np_proj], dim=-1)
############################################
# UTILIDAD: CREACIÓN DE DROPOUT PROGRESIVO
############################################
def progressive_dropout(p: float, depth: int) -> nn.Dropout:
    """
    Implementa un dropout progresivo que aumenta logarítmicamente con la profundidad.
    
    Args:
        p (float): Probabilidad base de dropout
        depth (int): Profundidad de la capa
        
    Returns:
        nn.Dropout: Módulo de dropout con probabilidad ajustada
    """
    if p == 0.0:
        return nn.Dropout(0.0)
    
    # Base logarítmica (ajustable según necesidades)
    base = 3.9
    
    # Usar logaritmo para un crecimiento más lento en capas profundas
    return nn.Dropout(p * (1 + math.log(depth + 1, base) * 0.04))
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
class HyperConnections(nn.Module):
    def __init__(self, d_model: int, expansion_rate: int = 4, dropout: float = 0.12, depth: int = 1):
        super().__init__()
        self.expansion_rate = expansion_rate
        
        # Determinar si CUDA está disponible
        device = torch.device('cuda')
        
        # Definición de las matrices estáticas - directamente en CUDA con bfloat16
        self.static_beta = nn.Parameter(torch.ones(expansion_rate, device=device, dtype=torch.bfloat16))
        
        # Inicialización de alpha según el paper - directamente en CUDA con bfloat16
        init_alpha0 = torch.zeros((expansion_rate, 1), device=device, dtype=torch.bfloat16)
        init_alpha0[depth % expansion_rate, 0] = 1.
        
        self.static_alpha = nn.Parameter(torch.cat(
            [init_alpha0, torch.eye(expansion_rate, device=device, dtype=torch.bfloat16)], dim=1))
        
        # Parámetros para la parte dinámica - directamente en CUDA con bfloat16
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros((d_model, expansion_rate+1), device=device, dtype=torch.bfloat16))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device, dtype=torch.bfloat16) * 0.01)
        self.dynamic_beta_fn = nn.Parameter(torch.zeros((d_model), device=device, dtype=torch.bfloat16))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device, dtype=torch.bfloat16) * 0.01)
        
        # Normalización para estabilidad
        self.layer_norm = nn.RMSNorm(d_model, eps=1e-5)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Pre-calcular buffers estáticos
        self.register_buffer(
            'static_alpha_expanded', 
            self.static_alpha.unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            'static_beta_expanded', 
            self.static_beta.unsqueeze(0).unsqueeze(0)
        )
    
    def _compute_dynamic_params(self, norm_x):
        """Calcular parámetros dinámicos (alpha y beta)"""
        dynamic_alpha = F.tanh(norm_x @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
        dynamic_beta = F.tanh(norm_x @ self.dynamic_beta_fn) * self.dynamic_beta_scale
        
        # Preparar para broadcasting
        dynamic_alpha = dynamic_alpha.unsqueeze(2)  # [B, T, 1, E+1]
        dynamic_beta = dynamic_beta.unsqueeze(2)    # [B, T, 1]
        
        # Combinar static y dynamic
        alpha = self.static_alpha_expanded + dynamic_alpha  # [B, T, E, E+1]
        beta = self.static_beta_expanded + dynamic_beta     # [B, T, E]
        
        return alpha, beta
    
    def _compute_width_connection(self, x, alpha):
        """Calcular la conexión de ancho (width connection)"""
        alpha_t = alpha.transpose(2, 3)  # [B, T, E+1, E]
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.expansion_rate, -1)  # [B, T, E, D]
        
        # Calcular mix_h con un solo einsum
        mix_h = torch.einsum('btij,btjd->btid', alpha_t, x_expanded)  # [B, T, E+1, D]
        return mix_h
    
    def _compute_depth_connection(self, residual, beta, mix_h):
        """Calcular la conexión de profundidad (depth connection) y combinar"""
        residual = self.dropout(residual)
        residual_expanded = residual.unsqueeze(2).expand(-1, -1, self.expansion_rate, -1)
        weighted_residual = residual_expanded * beta.unsqueeze(-1)  # [B, T, E, D]
        
        # Extraer mix_h_rest (todas excepto primera)
        mix_h_rest = mix_h[:, :, 1:, :]  # [B, T, E, D]
        
        # Combinar y reducir
        h = weighted_residual + mix_h_rest  # [B, T, E, D]
        output = h.sum(dim=2)  # [B, T, D]
        
        return output
    
    def forward(self, x, residual):
        """Forward pass con checkpointing para ahorrar memoria"""
        # Convertir las entradas a bfloat16 si no lo están ya
        x = x.to(dtype=torch.bfloat16)
        residual = residual.to(dtype=torch.bfloat16)
        
        # Paso 1: Normalizar entrada (no checkpointed - bajo uso de memoria)
        norm_x = self.layer_norm(x)
        
        # Función auxiliar para aplicar checkpoint y forzar el tipo de retorno
        def apply_checkpoint(func, *args):
            return cast(torch.Tensor, checkpoint.checkpoint(func, *args, use_reentrant=False))
        
        # Paso 2: Checkpoint para cálculo de parámetros dinámicos
        alpha, beta = apply_checkpoint(self._compute_dynamic_params, norm_x)
        
        # Paso 3: Checkpoint para width connection
        mix_h = apply_checkpoint(self._compute_width_connection, x, alpha)
        
        # Paso 4: Checkpoint para depth connection y combinación final
        output = apply_checkpoint(self._compute_depth_connection, residual, beta, mix_h)
        
        return output
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
    [MODIFICADO] Se eliminó el escalado ssmax_scale y seq_scale de Q.
    [MODIFICADO] Se aplica conversión explícita a bfloat16 *después* de las operaciones de normalización.
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
        # Añadido chequeo de divisibilidad para GQA
        elif num_heads % num_gqa_groups != 0:
            raise ValueError("num_heads debe ser divisible por num_gqa_groups")


        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            self.flash_attn_func = flash_attn_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
        except ImportError as e:
            # Mantener el comportamiento original de lanzar error si no se encuentra
            raise ImportError(f"Error al inicializar FlashAttention: {e}")

        # Para el unpadding
        try:
            from flash_attn.bert_padding import unpad_input, pad_input
            self.unpad_input = unpad_input
            self.pad_input = pad_input
        except ImportError as e:
            # Mantener el comportamiento original de lanzar error si no se encuentra
            raise ImportError(f"Error al importar funciones de padding: {e}")

        # Eliminada la inicialización de parámetros de escala ssmax_scale y seq_scale
        # self.ssmax_scale = nn.Parameter(torch.ones(num_heads, dtype=torch.bfloat16) * 0.168)
        # nn.init.uniform_(self.ssmax_scale, a=0.166, b=0.170)
        # self.register_buffer('seq_scale', torch.log(torch.tensor(max_seq_len, dtype=torch.bfloat16)))

        # Capas de normalización para la entrada (Pre-Norm en primer bloque o QKV-Norm para los demás)
        self.norm = nn.RMSNorm(embed_dim, eps=1e-5)

        # Capas de dropout (simplificadas)
        self.attention_dropout = progressive_dropout(dropout, depth=layer_index) # Usar layer_index
        # Eliminado: self.projection_dropout = progressive_dropout(dropout * 1.1, depth=1)
        self.output_dropout = progressive_dropout(dropout, depth=layer_index) # Usar layer_index

        # Proyecciones para Q, K, V usando GQAFANLinear (implementación FANformer)
        self.Wq = GQAFANLinear(embed_dim, embed_dim, num_heads, num_gqa_groups, p=p)
        self.Wk = GQAFANLinear(embed_dim, embed_dim, num_heads, num_gqa_groups, p=p)
        self.Wv = GQAFANLinear(embed_dim, embed_dim, num_heads, num_gqa_groups, p=p)

        # Proyección de salida (se mantiene como CoLA_Linear)
        self.out_proj = CoLA_Linear(embed_dim, embed_dim, rank=embed_dim // 4)

    def scaled_dot_product_attention_flash_unpadded(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                                   attention_mask: Optional[torch.Tensor] = None, # Revertido a Optional
                                                   is_causal: bool = False) -> torch.Tensor:
        B, H, S, D = q.shape  # batch, heads, sequence length, head dimension

        # Mantener la lógica original de manejo de máscara opcional
        if attention_mask is None:
            # Si no hay máscara de atención, usamos la versión regular
            return self.scaled_dot_product_attention_flash(q, k, v, mask=None, is_causal=is_causal)

        # Convertir las tensiones a [B, S, H, D] para unpad_input
        q_unpad = q.permute(0, 2, 1, 3)  # [B, S, H, D]
        k_unpad = k.permute(0, 2, 1, 3)  # [B, S, H, D]
        v_unpad = v.permute(0, 2, 1, 3)  # [B, S, H, D]

        # Preparar máscara: convertir a bool si es necesario
        # Mantener la lógica original
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        # Hacer unpadding de los tensores
        # Se mantienen las salidas originales, incluyendo el quinto elemento descartado
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

        # Eliminado el ajuste de q con factor de escala ssmax_scale y seq_scale
        # s = self.ssmax_scale.view(1, H, 1)
        # q_adjusted = q_norm * (self.seq_scale * s)

        # Factor de escala estándar para softmax

        try:
            # Usar flash attention sin padding, pasando q_norm
            output_unpadded = self.flash_attn_varlen_func(
                q_norm, k_norm, v_unpadded, # Usar q_norm directamente
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=self.attention_dropout.p,  # Aplicamos dropout aquí
                softmax_scale=None,         # Escala estándar
                causal=is_causal
            )

            # Volver a aplicar padding
            output_padded = self.pad_input(output_unpadded, indices_q, B, S)

            # Reorganizar a [B, H, S, D]
            output = output_padded.reshape(B, S, H, D).permute(0, 2, 1, 3)

            return output

        except Exception as e:
            # Mantener el manejo de errores original
            raise RuntimeError(f"Error en flash_attn_varlen_func: {e}")

    def scaled_dot_product_attention_flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                           mask: Optional[torch.Tensor] = None, # Mantener mask opcional
                                           is_causal: bool = False) -> torch.Tensor:
        # Normalizar vectores Q y K para mejorar estabilidad numérica
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # Eliminado el ajuste de q con factor de escala ssmax_scale y seq_scale
        # s = self.ssmax_scale.view(-1, 1, 1)
        # q_adjusted = q_norm * (self.seq_scale * s)

        # Preparar tensores para Flash Attention (requiere shape [B, S, H, D])
        q_trans = q_norm.permute(0, 2, 1, 3) # Usar q_norm directamente
        k_trans = k_norm.permute(0, 2, 1, 3)
        v_trans = v.permute(0, 2, 1, 3)

        # Mantener la verificación de dimensiones original
        if q_trans.size(-1) != k_trans.size(-1):
            raise ValueError(f"Las dimensiones de head no coinciden: q={q_trans.size(-1)}, k={k_trans.size(-1)}")

        # Factor de escala estándar para softmax
        try:
            # Aplicar Flash Attention, pasando q_trans
            output = self.flash_attn_func(
                q_trans, k_trans, v_trans,
                dropout_p=self.attention_dropout.p,  # Aplicamos dropout aquí
                softmax_scale=None,         # Escala estándar
                causal=is_causal
                # mask no se usa aquí
            )

            # Mantener la verificación de salida None original
            if output is None:
                raise ValueError("flash_attn_func devolvió None. Verifica las dimensiones y tipos de los tensores de entrada.")

            # Volver a la forma original
            output = output.permute(0, 2, 1, 3)
            return output

        except Exception as e:
            # Mantener el manejo de errores original
            raise RuntimeError(f"Error en flash_attn_func: {e}")

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, causal: bool = True) -> torch.Tensor:
        B, T, _ = X.shape
        norm_func = self.norm # Referencia a la capa de normalización

        # Implementación de HybridNorm*
        if self.use_pre_norm:
            # Primer bloque: Pre-Norm en atención
            # Aplicar norm y luego convertir explícitamente a bfloat16
            X_norm = norm_func(X)
            # Proyecciones para Q, K, V con FANformer
            Q = self.Wq(X_norm)  # [B, T, num_heads, head_dim]
            K = self.Wk(X_norm)  # [B, T, num_heads, head_dim]
            V = self.Wv(X_norm)  # [B, T, num_heads, head_dim]
        else:
            # Otros bloques: QKV-Norm
            # Aplicar norm y convertir explícitamente a bfloat16 antes de cada proyección
            Q = self.Wq(norm_func(X))
            K = self.Wk(norm_func(X))
            V = self.Wv(norm_func(X))

        # Permutar a formato [B, num_heads, T, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Aplicar RoPE si está activado
        if self.use_rope:
            Q = apply_rope_vectorized(Q)
            K = apply_rope_vectorized(K)

        # Convertir a bfloat16 para flash attention (mantener esta conversión explícita)
        Q = Q
        K = K
        V = V

        # Procesar la secuencia utilizando unpadding si hay máscara de atención
        # Mantener la lógica original para decidir la ruta
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

        # Eliminada la aplicación redundante de dropout (ya estaba eliminada)
        # attn_output = self.attention_dropout(attn_output)

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
        
        # Reemplazando GatedResidual con HyperConnections para atención
        self.hyper_conn_attn = HyperConnections(
            embed_dim, 
            expansion_rate=2,
            dropout=dropout, 
            depth=layer_index
        )
        
        # Post-Norm para FFN (HybridNorm)
        self.ffn_norm = nn.RMSNorm(embed_dim, eps=1e-5)
        self.mlp = SwiGLU(embed_dim, ff_dim, dropout, depth=1)
        
        # Reemplazando GatedResidual con HyperConnections para FFN
        self.hyper_conn_mlp = HyperConnections(
            embed_dim, 
            expansion_rate=2,
            dropout=dropout, 
            depth=layer_index
        )
        
        # Post-Norm final (HybridNorm)
        self.post_ffn_norm = nn.RMSNorm(embed_dim, eps=1e-5)

    def _attn_forward(self, x: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Parte de atención sin HyperConnections"""
        return self.attn(x, tgt_mask)
    
    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parte de feed-forward sin HyperConnections"""
        ffn_input = self.ffn_norm(x)
        return self.mlp(ffn_input)
    
    def _post_ffn_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalización final"""
        return self.post_ffn_norm(x)

    def forward(self, x: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward con checkpointing selectivo"""
        # Función auxiliar para aplicar checkpoint y forzar el tipo de retorno
        def apply_checkpoint(func, *args) -> torch.Tensor:
            # Usamos cast para indicar explícitamente al verificador de tipos
            # que el resultado de checkpoint.checkpoint es un tensor
            return cast(torch.Tensor, checkpoint.checkpoint(func, *args, use_reentrant=False))
        
        # Bloque de atención con HybridNorm
        if self.is_first_layer:
            # Primer bloque: Pre-Norm + QKV-Norm
            attention_output = apply_checkpoint(self._attn_forward, x, tgt_mask)
            attention_output = F.dropout(attention_output, p=self.hyper_conn_attn.dropout.p, training=self.training)
            hidden_states = self.hyper_conn_attn(x, attention_output)
        else:
            # Otros bloques: QKV-Norm
            attention_output = apply_checkpoint(self._attn_forward, x, tgt_mask)
            attention_output = F.dropout(attention_output, p=self.hyper_conn_attn.dropout.p, training=self.training)
            hidden_states = self.hyper_conn_attn(x, attention_output)
        
        # Paso 3: Aplicar checkpoint al feed-forward
        ffn_output = apply_checkpoint(self._ffn_forward, hidden_states)
        
        # Aplicar dropout a la salida de FFN
        ffn_output = F.dropout(ffn_output, p=self.hyper_conn_mlp.dropout.p, training=self.training)
        
        # Paso 4: Aplicar HyperConnections
        hidden_states = self.hyper_conn_mlp(hidden_states, ffn_output)
        
        # Paso 5: Aplicar checkpoint a la normalización final
        output = apply_checkpoint(self._post_ffn_norm_forward, hidden_states)
        
        return output
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
    Versión simplificada con skip connections directas sin gates.
    """
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.12,
                 num_gqa_groups: Optional[int] = None, p: float = 0.15,
                 use_checkpoint: bool = True, skip_every: int = 3):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.skip_every = skip_every
        self.embed_dim = embed_dim

        # Crear capas de FANformer con tratamiento especial para el primer bloque (HybridNorm*)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_first_layer = (i == 0)  # Identificar si es el primer bloque para HybridNorm*
            self.layers.append(
                FANformerLayer(
                    embed_dim, num_heads, ff_dim,
                    dropout=dropout * (1 + i * 0.035),
                    layer_index=i+1,
                    num_gqa_groups=num_gqa_groups,
                    is_first_layer=is_first_layer,
                    p=p
                )
            )

        num_skips = num_layers // skip_every
        
        # Mantenemos los dropouts pero eliminamos los gates y normalizaciones
        self.skip_dropouts = nn.ModuleList([
            progressive_dropout(dropout * 0.8, depth=i+1)
            for i in range(num_skips)
        ])

        # Mantenemos las normalizaciones finales
        self.dropout = progressive_dropout(dropout, depth=1)
        self.layer_norm = nn.RMSNorm(embed_dim, eps=1e-5)
            
    def forward(self, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        layer_states = []

        for i, layer in enumerate(self.layers):
            if i % self.skip_every == 0:
                layer_states.append(output)

            # Añadimos cuda empty cada 4 capas
            if i > 0 and i % 4 == 0:
                torch.cuda.empty_cache()

            # Simplemente llamamos al método forward estándar
            output = layer(output, tgt_mask)

            if (i + 1) % self.skip_every == 0 and i // self.skip_every < len(self.skip_dropouts):
                skip_idx = i // self.skip_every

                # Obtener skip state
                skip_state = layer_states[skip_idx]

                # Aplicar dropout directamente (sin normalización ni gates)
                skip_state_dropped = self.skip_dropouts[skip_idx](skip_state)

                # Combinar directamente sin gates
                output = output + skip_state_dropped

        # Normalizaciones finales
        output = self.dropout(output)
        output = self.layer_norm(output)

        return output
############################################
# MODELO TEXT-ONLY (DECODER-ONLY)
############################################
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int,
                 num_heads: int, num_decoder_layers: int, ff_dim: int, dropout: float = 0.12,
                 num_gqa_groups: Optional[int] = None, p: float = 0.15,
                 tie_weights: bool = True):
        super().__init__()
        self.epsilon = 1e-5
        self.dropout_rate = dropout
        self.text_feat_norm = nn.RMSNorm(embed_dim, eps=self.epsilon)
        print("[DEBUG] MultiModalModel: modo decoder-only activado")
        
        # Matriz de embedding
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        init_embedding(self.decoder_embedding)
        self.emb_dropout = progressive_dropout(self.dropout_rate, depth=1)
        self.decoder_input_norm = nn.RMSNorm(embed_dim, eps=self.epsilon)
        
        print(f"[DEBUG] Usando FANformer con p={p}")
        self.decoder = FANformerDecoder(
            num_decoder_layers, embed_dim, num_heads, ff_dim, 
            dropout=self.dropout_rate, num_gqa_groups=num_gqa_groups, p=p
        )
        
        # Crear una proyección de salida lineal simple en lugar de CoLA_Linear
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Compartir pesos entre embeddings y proyección final
        if tie_weights:
            print("[DEBUG] Compartiendo pesos entre matriz de embedding y proyección de salida")
            self.output_proj.weight = self.decoder_embedding.weight
        
        self.tokenizer = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        dec_emb = self.decoder_embedding(tokens)
        dec_emb = self.emb_dropout(dec_emb)
        dec_emb = self.decoder_input_norm(dec_emb)
        dec_out = self.decoder(dec_emb)
        logits = self.output_proj(dec_out)
        return logits
    def sample_next_token(self, logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
        """
        Muestrea el siguiente token usando temperatura, top-k y top-p.
        Elimina la lógica de beam search y penalización por repetición.
        """
        B, V = logits.shape # Batch size, Vocab size

        # 1. Aplicar Temperatura
        # Evitar división por cero o temperatura negativa
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature

        # 2. Aplicar Top-K
        if top_k > 0:
            # Obtener los k valores más altos
            # min(top_k, V) para manejar casos donde k > vocab_size
            values, _ = torch.topk(logits, min(top_k, V))
            # Obtener el valor del k-ésimo token
            kth_value = values[:, -1].unsqueeze(1)
            # Poner a -inf todos los logits menores que el k-ésimo valor
            logits[logits < kth_value] = -float('inf')

        # 3. Aplicar Top-P (Nucleus Sampling)
        if top_p > 0.0:
            # Ordenar los logits en orden descendente
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # Calcular probabilidades acumuladas
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Crear una máscara para los tokens a eliminar (aquellos fuera del núcleo p)
            # Encontrar los índices que exceden la probabilidad acumulada p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Importante: Asegurarse de mantener al menos el token con la probabilidad más alta
            # Desplazar la máscara a la derecha para que el primer elemento nunca se elimine
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Crear la máscara final en el orden original de los logits
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            # Usar scatter_ para poner True en las posiciones a eliminar
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

            # Aplicar la máscara, poniendo los logits a -inf
            logits[indices_to_remove] = -float('inf')

        # 4. Muestrear desde la distribución modificada
        # Convertir logits a probabilidades
        probs = F.softmax(logits, dim=-1)
        # Muestrear un token basado en las probabilidades
        # `num_samples=1` para obtener un token por secuencia en el batch
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def forward_inference(self, prompt_tokens: torch.Tensor, max_length: int = 100, min_length: int = 1,
                          top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """
        Genera texto usando muestreo (temperatura, top-k, top-p) sin beam search ni penalización.
        """
        if self.tokenizer is None:
            raise ValueError("El tokenizador no ha sido asignado al modelo. Asigna 'model.tokenizer = tokenizer'.")

        # Eliminar el historial y las puntuaciones de beam search, ya no son necesarios
        # self._generated_history = [] # Eliminado
        # self._beam_scores = None # Eliminado

        generated = prompt_tokens
        cur_length = generated.size(1)

        # Bucle de generación
        for _ in range(max_length): # `max_length` controla los tokens *adicionales* a generar
            # Preparar entrada para el modelo (últimos tokens generados)
            # No es necesario truncar aquí si el modelo maneja secuencias largas,
            # pero si hay límites estrictos, podrías necesitar `generated[:, -self.model.max_seq_len:]`
            input_tokens = generated

            # Forward pass del modelo
            dec_emb = self.decoder_embedding(input_tokens)
            dec_emb = self.emb_dropout(dec_emb)
            dec_emb = self.decoder_input_norm(dec_emb)
            dec_out = self.decoder(dec_emb)
            logits = self.output_proj(dec_out)

            # Obtener los logits solo para el *último* token de la secuencia
            next_token_logits = logits[:, -1, :]

            # Muestrear el siguiente token usando la función simplificada
            next_token = self.sample_next_token(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )

            # Actualizar la secuencia generada
            generated = torch.cat([generated, next_token], dim=1)
            cur_length += 1

            # Condición de parada: longitud mínima alcanzada Y se generó el token EOS
            # Añadir chequeo para self.tokenizer.eos_token_id no ser None
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_token_id is not None and cur_length > min_length and (next_token == eos_token_id).all():
                 break # Detener si se genera EOS después de la longitud mínima

        # Eliminar limpieza de historial/scores
        # del self._generated_history # Eliminado
        # self._beam_scores = None # Eliminado

        return generated


############################################
# MÓDULOS DE DATOS: SOLO TEXTO
############################################

class TextCleaningProcessor:
    """
    Clase encargada de la limpieza y procesamiento de los datos textuales.
    Maneja la carga, limpieza, shuffle y estratificación de los datos.
    """
    def __init__(self, max_examples: int, fineweb_ratio: float = 0.9, 
                 val_split_ratio: float = 0.1, seed: int = 42):
        self.max_examples = max_examples
        self.fineweb_ratio = fineweb_ratio
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        
        # Calcular cuántos ejemplos cargar de cada fuente basado en el ratio
        self.max_fineweb_examples = int(self.max_examples * self.fineweb_ratio)
        self.max_math_examples = self.max_examples - self.max_fineweb_examples
        
        # Información sobre el dataset
        self.dataset_info = {
            "source": "HuggingFaceFW/fineweb-edu",
            "additional_source": "HuggingFaceTB/finemath",
            "original_license": "ODC-By v1.0",
            "terms_of_use": "CommonCrawl Terms of Use",
            "modifications": "Text cleaning applied, removed technical elements",
            "mix_ratio": f"{self.fineweb_ratio*100:.0f}% FineWeb, {(1-self.fineweb_ratio)*100:.0f}% FineMath"
        }
    
    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando URLs y elementos técnicos."""
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'http[s]?://(?:[a-zA-Z0-9./?=&_-]+)', '', text)
        text = re.sub(r's3://commoncrawl/crawl-data/.*\n?', '', text)
        text = re.sub(r'CC-MAIN-\d{4}-\d{2}', '', text)
        text = re.sub(r'<urn:uuid:[a-f0-9-]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_math_text(self, text: str) -> str:
        """Limpia el texto del dataset de matemáticas."""
        if text is None:
            return ""
            
        # Eliminar URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Eliminar caracteres extraños y normalizar espacio en blanco
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar referencias como "www.ejemplo.com"
        text = re.sub(r'www\.\S+\.\w+', '', text)
        
        # Corregir espacios antes de signos de puntuación
        text = re.sub(r'\s([.,;:!?])', r'\1', text)
        
        # Eliminar referencias a archivos o rutas
        text = re.sub(r'[a-zA-Z0-9_\-\.]+\.(pdf|cgi)', '', text)
        
        # Eliminar paréntesis vacíos o con un solo carácter
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\(\s*.\s*\)', '', text)
        
        # Normalizar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_streaming_data(self, clean_data_dir: Path) -> Path:
        """
        Procesa datos de FineWeb y FineMath usando streaming.
        Limpia los textos y guarda los resultados en la carpeta clean_data.
        """
        clean_data_file = clean_data_dir / "cleaned_texts.jsonl"
        
        # Si ya existe el archivo de datos limpios, saltamos este paso
        if clean_data_file.exists():
            print(f"Archivo de datos limpios encontrado: {clean_data_file}")
            return clean_data_file
            
        print(f"Procesando y limpiando datos (objetivo: {self.max_examples} ejemplos)...")
        
        # Cargar FineWeb con streaming
        fineweb_stream = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name="sample-10BT", 
            split="train", 
            streaming=True
        )
        
        # Cargar FineMath con streaming
        finemath_stream = load_dataset(
            "HuggingFaceTB/finemath", 
            "finemath-4plus", 
            split="train", 
            streaming=True
        )
        
        # Contador de ejemplos procesados y limpios
        fineweb_count = 0
        finemath_count = 0
        
        # Crear archivo JSONL para guardar textos limpios
        with open(clean_data_file, 'w', encoding='utf-8') as f:
            # Procesar FineWeb primero
            print(f"Procesando FineWeb (objetivo: {self.max_fineweb_examples} ejemplos)...")
            for i, example in enumerate(tqdm(fineweb_stream)):
                if fineweb_count >= self.max_fineweb_examples:
                    break
                    
                text = example.get("text", "")
                cleaned_text = self.clean_text(text)
                
                # Filtrar textos muy cortos o vacíos
                if cleaned_text and len(cleaned_text.split()) > 10:
                    example_data = {
                        "text": cleaned_text,
                        "domain": "fineweb",
                        "id": f"fineweb_{fineweb_count}"
                    }
                    f.write(json.dumps(example_data) + '\n')
                    fineweb_count += 1
                    
                if (i + 1) % 10000 == 0:
                    print(f"Procesados {i+1} ejemplos de FineWeb, limpios: {fineweb_count}")
            
            # Procesar FineMath 
            print(f"Procesando FineMath (objetivo: {self.max_math_examples} ejemplos)...")
            for i, example in enumerate(tqdm(finemath_stream)):
                if finemath_count >= self.max_math_examples:
                    break
                    
                text = example.get("text", "")
                cleaned_text = self.clean_math_text(text)
                
                # Filtrar textos muy cortos o vacíos
                if cleaned_text and len(cleaned_text.split()) > 10:
                    example_data = {
                        "text": cleaned_text,
                        "domain": "math",
                        "id": f"math_{finemath_count}"
                    }
                    f.write(json.dumps(example_data) + '\n')
                    finemath_count += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"Procesados {i+1} ejemplos de FineMath, limpios: {finemath_count}")
        
        print(f"Procesamiento completado:")
        print(f"  - Ejemplos de FineWeb limpios: {fineweb_count}")
        print(f"  - Ejemplos de FineMath limpios: {finemath_count}")
        print(f"  - Total: {fineweb_count + finemath_count}")
        
        # Guardar información de atribución
        attribution_info = {
            "source": self.dataset_info["source"],
            "additional_source": self.dataset_info["additional_source"],
            "original_license": self.dataset_info["original_license"],
            "terms_of_use": self.dataset_info["terms_of_use"],
            "modifications": self.dataset_info["modifications"],
            "mix_ratio": f"{fineweb_count/(fineweb_count+finemath_count)*100:.1f}% FineWeb, "
                       f"{finemath_count/(fineweb_count+finemath_count)*100:.1f}% FineMath",
            "processed_examples": fineweb_count + finemath_count,
            "processing_date": str(datetime.now()),
            "min_words_filter": 10
        }
        
        with open(clean_data_dir / 'dataset_attribution.json', 'w') as f:
            json.dump(attribution_info, f, indent=2)
            
        return clean_data_file

    def shuffle_data(self, clean_data_file: Path, shuffled_dir: Path) -> Path:
        """
        Realiza un shuffle de los datos limpios.
        """
        shuffled_file = shuffled_dir / "shuffled_data.jsonl"
        
        # Si ya existe el archivo de datos barajados, saltamos este paso
        if shuffled_file.exists():
            print(f"Archivo de datos barajados encontrado: {shuffled_file}")
            return shuffled_file
            
        print("Realizando shuffle de los datos limpios...")
        
        # Cargar todos los ejemplos en memoria para el shuffle
        examples = []
        with open(clean_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Shuffle usando NumPy con la semilla especificada
        np.random.seed(self.seed)
        indices = np.random.permutation(len(examples))
        shuffled_examples = [examples[i] for i in indices]
        
        # Guardar ejemplos barajados
        with open(shuffled_file, 'w', encoding='utf-8') as f:
            for example in shuffled_examples:
                f.write(json.dumps(example) + '\n')
                
        print(f"Shuffle completado, {len(shuffled_examples)} ejemplos guardados en {shuffled_file}")
        return shuffled_file

    def stratify_data(self, shuffled_file: Path, stratified_dir: Path) -> tuple[Path, Path]:
        """
        Realiza una estratificación de los datos usando sklearn.
        Divide los datos en conjuntos de entrenamiento y validación,
        manteniendo la proporción de dominios.
        """
        train_file = stratified_dir / "train_data.jsonl"
        val_file = stratified_dir / "val_data.jsonl"
        
        # Si ya existen los archivos de datos estratificados, saltamos este paso
        if train_file.exists() and val_file.exists():
            print(f"Archivos de datos estratificados encontrados: {train_file} y {val_file}")
            return train_file, val_file
            
        print("Realizando estratificación de los datos...")
        
        # Cargar todos los ejemplos
        examples = []
        with open(shuffled_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Extraer dominios para estratificación
        domains = [example['domain'] for example in examples]
        
        # Crear división estratificada con sklearn
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.val_split_ratio,
            random_state=self.seed
        )
        
        # Obtener índices de train y val
        train_indices, val_indices = next(splitter.split(examples, domains))
        
        # Dividir los datos según los índices
        train_examples = [examples[i] for i in train_indices]
        val_examples = [examples[i] for i in val_indices]
        
        # Guardar ejemplos de entrenamiento
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
                
        # Guardar ejemplos de validación
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example) + '\n')
        
        # Calcular estadísticas
        train_fineweb = sum(1 for ex in train_examples if ex['domain'] == 'fineweb')
        train_math = sum(1 for ex in train_examples if ex['domain'] == 'math')
        val_fineweb = sum(1 for ex in val_examples if ex['domain'] == 'fineweb')
        val_math = sum(1 for ex in val_examples if ex['domain'] == 'math')
        
        print(f"Estratificación completada:")
        print(f"Entrenamiento: {len(train_examples)} ejemplos totales")
        print(f"  - FineWeb: {train_fineweb} ({train_fineweb/len(train_examples)*100:.1f}%)")
        print(f"  - FineMath: {train_math} ({train_math/len(train_examples)*100:.1f}%)")
        print(f"Validación: {len(val_examples)} ejemplos totales")
        print(f"  - FineWeb: {val_fineweb} ({val_fineweb/len(val_examples)*100:.1f}%)")
        print(f"  - FineMath: {val_math} ({val_math/len(val_examples)*100:.1f}%)")
        
        return train_file, val_file

    def get_domain_stats(self, stratified_dir: Path) -> dict:
        """
        Analiza y devuelve estadísticas de distribución por dominio.
        Lee directamente desde los archivos de train y val.
        """
        train_file = stratified_dir / "train_data.jsonl"
        val_file = stratified_dir / "val_data.jsonl"
        
        # Verificar que los archivos existan
        if not train_file.exists() or not val_file.exists():
            raise ValueError("Los archivos de datos estratificados no existen.")
            
        # Analizar archivos de train
        train_examples = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                train_examples.append(json.loads(line))
        
        # Analizar archivos de val
        val_examples = []
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                val_examples.append(json.loads(line))
        
        # Calcular estadísticas
        train_fineweb = sum(1 for ex in train_examples if ex['domain'] == 'fineweb')
        train_math = sum(1 for ex in train_examples if ex['domain'] == 'math')
        train_total = len(train_examples)
        
        val_fineweb = sum(1 for ex in val_examples if ex['domain'] == 'fineweb')
        val_math = sum(1 for ex in val_examples if ex['domain'] == 'math')
        val_total = len(val_examples)
        
        return {
            "train": {
                "fineweb_percent": train_fineweb / train_total * 100 if train_total else 0,
                "math_percent": train_math / train_total * 100 if train_total else 0,
                "fineweb_count": train_fineweb,
                "math_count": train_math
            },
            "validation": {
                "fineweb_percent": val_fineweb / val_total * 100 if val_total else 0,
                "math_percent": val_math / val_total * 100 if val_total else 0,
                "fineweb_count": val_fineweb,
                "math_count": val_math
            }
        }
class CountedWebDataset(wds.WebDataset):
    def __init__(self, urls, length, **kwargs):
        super().__init__(urls, **kwargs)
        self.length = length
        
    def __len__(self):
        return self.length
class TextTokenizer:
    """
    Clase encargada de la tokenización y creación de WebDatasets.
    Maneja la conversión de texto a tokens y la creación de archivos TAR.
    """
    def __init__(self, tokenizer, max_seq_len: int, stride: int = 16, 
                 shard_size: int = 1000, num_workers: int = 16):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.shard_size = shard_size
        self.num_workers = min(32, num_workers)  # Limitar a un máximo de 16 workers
        self.num_examples = 0  # Inicializar el atributo
    
    # Función estática para ser pickleable
    @staticmethod
    def identity_fn(x):
        """Función de identidad serializable para reemplazar las lambdas."""
        return x
    
    def _save_example_count(self, prefix, count):
        """Guarda el número de ejemplos en un archivo."""
        count_file = Path(prefix).parent / f"{Path(prefix).name}_count.txt"
        with open(count_file, 'w') as f:
            f.write(str(count))
        print(f"Guardado número de ejemplos: {count} en {count_file}")

    def _load_example_count(self, prefix):
        """Carga el número de ejemplos desde un archivo."""
        count_file = Path(prefix).parent / f"{Path(prefix).name}_count.txt"
        if count_file.exists():
            with open(count_file, 'r') as f:
                return int(f.read().strip())
        return None
    def tokenize_and_create_webdataset(self, train_file: Path, val_file: Path, 
                                     webdataset_dir: Path) -> tuple[str, str]:
        """
        Tokeniza los datos y los convierte al formato WebDataset (archivos TAR).
        Utiliza la API oficial de datasets de Hugging Face para la paralelización.
        """
        train_prefix = webdataset_dir / "train"
        val_prefix = webdataset_dir / "val"
        
        # Verificar si ya existen archivos WebDataset
        train_shards = list(webdataset_dir.glob("train-*.tar"))
        val_shards = list(webdataset_dir.glob("val-*.tar"))
        
        if train_shards and val_shards:
            print(f"Archivos WebDataset encontrados: {len(train_shards)} train shards, {len(val_shards)} val shards")
            
            # Cargar count desde archivo
            saved_count = self._load_example_count(train_prefix)
            if saved_count is not None:
                self.num_examples = saved_count
                print(f"Cargado número de ejemplos desde cache: {self.num_examples}")
            else:
                # Estimar conteo basado en shards y tamaño de shard
                self.num_examples = len(train_shards) * self.shard_size
                print(f"Número estimado de ejemplos (no se encontró cache): {self.num_examples}")
                # Guardar para futuras ejecuciones
                self._save_example_count(train_prefix, self.num_examples)
                
            return str(train_prefix), str(val_prefix)
            
        print("Tokenizando datos y creando archivos WebDataset...")
        
        # Procesar archivos de train y val
        train_shards = self._process_file_to_webdataset(train_file, train_prefix)
        # Guardar el contador de ejemplos después de procesar train
        self._save_example_count(train_prefix, self.num_examples)
        train_examples = self.num_examples
        
        val_shards = self._process_file_to_webdataset(val_file, val_prefix)
        # Restaurar el contador para train después de procesar val
        self.num_examples = train_examples
        
        print(f"WebDataset creado:")
        print(f"  - Train: {train_shards} shards con {self.num_examples} ejemplos")
        print(f"  - Val: {val_shards} shards")
        
        return str(train_prefix), str(val_prefix)  # Convertir a string para evitar problemas con Path
    
    def _tokenize_function(self, example):
        """Función para tokenizar un ejemplo"""
        # Tokenizar el texto con soporte para overflowing tokens
        encodings = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
            return_overflowing_tokens=True,
            stride=self.stride
        )
        
        # Convertir tensores a listas para serialización
        input_ids_list = [ids.tolist() for ids in encodings["input_ids"]]
        attention_mask_list = [mask.tolist() for mask in encodings["attention_mask"]] if "attention_mask" in encodings else None
        
        # Crear un nuevo ejemplo para cada fragmento tokenizado
        result = {
            "original_id": [example["id"]] * len(input_ids_list),
            "domain": [example["domain"]] * len(input_ids_list),
            "fragment_idx": list(range(len(input_ids_list))),
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list if attention_mask_list else [None] * len(input_ids_list)
        }
        
        return result
    
    def _create_shard(self, examples, output_prefix, shard_idx):
        """Función para crear un shard WebDataset"""
        tar_filename = f"{output_prefix}-{shard_idx:05d}.tar"
        with tarfile.open(tar_filename, "w") as tar:
            for i, example in enumerate(examples):
                fragment_id = f"{shard_idx:05d}_{i:05d}"
                
                # Crear metadatos
                metadata = {
                    "original_id": example["original_id"],
                    "domain": example["domain"],
                    "fragment_idx": example["fragment_idx"]
                }
                json_content = json.dumps(metadata).encode('utf-8')
                
                # Crear contenido de input_ids y attention_mask
                input_ids_content = json.dumps(example["input_ids"]).encode('utf-8')
                attention_mask_content = json.dumps(example["attention_mask"]).encode('utf-8') if example["attention_mask"] is not None else None
                
                # Añadir archivos al TAR
                json_info = tarfile.TarInfo(f"{fragment_id}.json")
                json_info.size = len(json_content)
                tar.addfile(json_info, io.BytesIO(json_content))
                
                input_ids_info = tarfile.TarInfo(f"{fragment_id}.input.json")
                input_ids_info.size = len(input_ids_content)
                tar.addfile(input_ids_info, io.BytesIO(input_ids_content))
                
                if attention_mask_content:
                    attn_info = tarfile.TarInfo(f"{fragment_id}.mask.json")
                    attn_info.size = len(attention_mask_content)
                    tar.addfile(attn_info, io.BytesIO(attention_mask_content))

    def _process_file_to_webdataset(self, input_file, output_prefix):
        """Función para procesar un archivo completo de forma más eficiente en memoria."""
        print(f"Cargando ejemplos de {input_file}")
        dataset = HFDataset.from_json(str(input_file))
        num_original_examples = len(dataset) # Guardar número original
        print(f"Cargados {num_original_examples} ejemplos originales")

        print(f"Tokenizando con {self.num_workers} procesos en paralelo...")
        # El map() de datasets ya muestra una barra de progreso para la tokenización
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=False, # O True si _tokenize_function maneja lotes
            num_proc=self.num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizando datos"
        )
        print("Tokenización completada. Iniciando escritura de shards...")

        shard_buffer = []
        shard_idx = 0
        fragment_count_total = 0
        fragments_in_shard_buffer = 0

        # --- Modificación para la barra de progreso ---
        # Estimamos el número total de lotes para la barra de progreso del iterador
        batch_size_iterator = 500 # El tamaño de lote que usas en .iter()
        try:
            # Intentamos obtener una longitud del dataset tokenizado
            # Nota: Esto puede no ser preciso si .map cambió el número de elementos (no debería si batched=False)
            # O puede no ser soportado si es un iterable muy perezoso.
            total_items_for_iterator = len(tokenized_dataset)
            total_batches = (total_items_for_iterator + batch_size_iterator - 1) // batch_size_iterator
        except TypeError:
            total_batches = None # No podemos determinar el total
            print("[Advertencia] No se pudo determinar el número total de lotes para la barra de progreso.")

        iterator = tokenized_dataset.iter(batch_size=batch_size_iterator)

        # Crear el objeto tqdm manualmente para tener más control
        pbar = tqdm(iterator, total=total_batches, desc="Procesando fragmentos", unit="batch")
        # ---------------------------------------------

        for batch in pbar: # Iterar sobre el objeto tqdm
            batch_fragments = []
            num_examples_in_batch = len(batch['original_id']) # Asumiendo que las claves tienen la misma longitud de lote

            for i in range(num_examples_in_batch):
                # Asegurarse de que accedemos correctamente si _tokenize_function devolvió listas anidadas
                # Si _tokenize_function devuelve una estructura plana por fragmento, esto necesita ajuste
                # Asumiendo la estructura original donde cada ejemplo tiene una lista de fragmentos:
                if isinstance(batch["input_ids"][i], list) and isinstance(batch["input_ids"][i][0], list):
                     num_fragments_in_example = len(batch["input_ids"][i])
                     for j in range(num_fragments_in_example):
                         fragment = {
                            "original_id": batch["original_id"][i][j],
                            "domain": batch["domain"][i][j],
                            "fragment_idx": batch["fragment_idx"][i][j],
                            "input_ids": batch["input_ids"][i][j],
                            "attention_mask": batch["attention_mask"][i][j]
                         }
                         batch_fragments.append(fragment)
                else:
                     # Caso donde _tokenize_function ya aplanó o no hubo fragmentación
                     # Esto podría necesitar un ajuste basado en la salida exacta de _tokenize_function
                     num_fragments_in_example = 1 # Asumir 1 si no es una lista de listas
                     fragment = {
                        "original_id": batch["original_id"][i],
                        "domain": batch["domain"][i],
                        "fragment_idx": batch["fragment_idx"][i], # Puede ser siempre 0 o una lista
                        "input_ids": batch["input_ids"][i],
                        "attention_mask": batch["attention_mask"][i]
                     }
                     # Ajustar índices si son listas [0] en lugar de valores directos
                     if isinstance(fragment["fragment_idx"], list): fragment["fragment_idx"] = fragment["fragment_idx"][0]
                     if isinstance(fragment["original_id"], list): fragment["original_id"] = fragment["original_id"][0]
                     if isinstance(fragment["domain"], list): fragment["domain"] = fragment["domain"][0]

                     batch_fragments.append(fragment)


            # Añadir fragmentos del batch al buffer del shard
            fragments_processed_in_batch = 0
            for fragment in batch_fragments:
                shard_buffer.append(fragment)
                fragments_in_shard_buffer += 1
                fragment_count_total += 1
                fragments_processed_in_batch += 1

                # Si el buffer está lleno, escribir el shard
                if fragments_in_shard_buffer >= self.shard_size:
                    # print(f"Escribiendo shard {shard_idx} con {len(shard_buffer)} fragmentos...") # Opcional, tqdm ya informa
                    self._create_shard(shard_buffer, output_prefix, shard_idx)
                    shard_idx += 1
                    shard_buffer = []  # Vaciar buffer
                    fragments_in_shard_buffer = 0

            # --- Actualizar la barra de progreso ---
            # Actualiza el postfijo para mostrar el recuento total de fragmentos
            pbar.set_postfix({"Fragments": f"{fragment_count_total:,}"})
            # pbar.update(1) # tqdm(iterator) ya maneja la actualización por lote
            # ---------------------------------------

        # Cerrar la barra de progreso al finalizar el bucle
        pbar.close()

        # Escribir el último shard si quedan fragmentos en el buffer
        if shard_buffer:
            print(f"Escribiendo último shard {shard_idx} con {len(shard_buffer)} fragmentos...")
            self._create_shard(shard_buffer, output_prefix, shard_idx)
            num_shards = shard_idx + 1
        else:
            num_shards = shard_idx

        print(f"Proceso completado: {fragment_count_total:,} fragmentos generados en {num_shards} shards.") # Añadir formato
        self.num_examples = fragment_count_total

        return num_shards
    def create_webdataset_datapipeline(self, prefix, shuffle_buffer_size: int = 1000, is_train=True):
        """
        Crea un pipeline de datos basado en WebDataset.
        """
        # Construir el patrón de URL para los shards
        prefix_path = Path(prefix)
        shard_files = sorted(list(prefix_path.parent.glob(f"{prefix_path.name}-*.tar")))
        if not shard_files:
            raise FileNotFoundError(f"No se encontraron archivos de shard para el patrón {prefix}-*.tar")
            
        # Crear una lista de URLs absolutas para cada archivo
        urls = [str(f.absolute()) for f in shard_files]
        print(f"Cargando {len(urls)} shards de {prefix}")
        
        # Ajustar el número de nodos en función de los shards disponibles
        # Usar la clase CountedWebDataset definida fuera del método
        dataset = CountedWebDataset(
            urls, 
            length=self.num_examples,
            shardshuffle=len(shard_files) if is_train else 0,
            nodesplitter=None,
            handler=wds.handlers.warn_and_continue,
            empty_check=False  # Añadir este parámetro para evitar el error
        )
        
        # Reducir los tamaños de buffer para limitar el uso de memoria
        if is_train:
            dataset = dataset.shuffle(shuffle_buffer_size // 2)  # Buffer más pequeño
            dataset = dataset.map(self._decode_example)
            dataset = dataset.batched(500, partial=True)
            dataset = dataset.shuffle(1000)
            dataset = dataset.unbatched()
            
            # Usar funciones pickleables
            dataset = dataset.map_dict(
                text=self.identity_fn,
                domain=self.identity_fn,
                original_id=self.identity_fn
            )
        else:
            dataset = dataset.map(self._decode_example)
        
        return dataset
    def _decode_example(self, sample):
        """Función para decodificar y preparar los ejemplos"""
        # Extraer y convertir datos
        metadata = json.loads(sample["json"].decode("utf-8"))
        input_ids = torch.tensor(json.loads(sample["input.json"].decode("utf-8")))
        
        result = {
            "text": input_ids,
            "domain": metadata["domain"],
            "original_id": metadata["original_id"]
        }
        
        # Añadir máscara de atención si existe
        if "mask.json" in sample:
            attention_mask = torch.tensor(json.loads(sample["mask.json"].decode("utf-8")))
            result["attention_mask"] = attention_mask
            
        return result

class FinewebTextDataModule(pl.LightningDataModule):
    """
    Módulo de datos para cargar y procesar textos combinando FineWeb y FineMath.
    Optimizado para procesar textos largos usando WebDataset y estratificación con sklearn.
    
    Implementa el siguiente flujo:
    1. Limpieza de datos (streaming) y guardado
    2. Shuffle y guardado
    3. Estratificación, shuffle y guardado
    4. Tokenización y guardado
    5. Uso de WebDataset para dataloaders
    """
    def __init__(self, tokenizer, max_seq_len: int, batch_size: int, max_examples: int = 10000, 
                 val_split_ratio: float = 0.1, stride: int = 16, 
                 fineweb_ratio: float = 0.9, seed: int = 42, token: str = None,
                 shuffle_buffer_size: int = 1000, shard_size: int = 40000,
                 num_workers: int = 32):
        """
        Inicializa el módulo de datos.
        
        Args:
            tokenizer: Tokenizador a utilizar
            max_seq_len: Longitud máxima de secuencia para tokenización
            batch_size: Tamaño del lote para carga de datos
            max_examples: Número máximo TOTAL de ejemplos (FineWeb + FineMath)
            val_split_ratio: Proporción de ejemplos para validación
            stride: Solapamiento entre fragmentos consecutivos
            fineweb_ratio: Proporción de ejemplos de FineWeb en el dataset final
            seed: Semilla para reproducibilidad
            token: Token de HuggingFace para acceder a datasets protegidos
            shuffle_buffer_size: Tamaño del buffer para shuffle de WebDataset
            shard_size: Número de ejemplos por shard en WebDataset
            num_workers: Número máximo de procesos para tokenización paralela (máximo 16)
        """
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.token = token
        
        # Inicializar los directorios para datos intermedios
        self.base_dir = Path(f"dataset_{max_examples}")
        self.clean_data_dir = self.base_dir / "clean_data"
        self.shuffled_dir = self.base_dir / "shuffled"
        self.stratified_dir = self.base_dir / "stratified"
        self.tokenized_dir = self.base_dir / "tokenized"
        self.webdataset_dir = self.base_dir / "webdataset"
        
        # Crear directorios si no existen
        for directory in [self.base_dir, self.clean_data_dir, self.shuffled_dir, 
                         self.stratified_dir, self.tokenized_dir, self.webdataset_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Inicializar las clases auxiliares
        self.data_cleaner = TextCleaningProcessor(
            max_examples=max_examples,
            fineweb_ratio=fineweb_ratio,
            val_split_ratio=val_split_ratio,
            seed=seed
        )
        
        self.tokenizer_processor = TextTokenizer(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            stride=stride,
            shard_size=shard_size,
            num_workers=num_workers
        )
        
        # Atributos para datasets
        self.train_dataset = None
        self.val_dataset = None
        
        # Valor de padding
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 100349

    def setup(self, stage: Optional[str] = None):
        """
        Prepara los datos para entrenamiento y validación.
        Implementa el flujo completo: limpieza, shuffle, estratificación, tokenización, WebDataset.
        """
        print(f"Configurando dataset con máximo {self.data_cleaner.max_examples} ejemplos totales.")
        print(f"Distribución objetivo: {self.data_cleaner.fineweb_ratio*100:.0f}% FineWeb, "
              f"{(1-self.data_cleaner.fineweb_ratio)*100:.0f}% FineMath")
        
        # Paso 1: Procesar y limpiar datos
        clean_data_file = self.data_cleaner.process_streaming_data(self.clean_data_dir)
        
        # Paso 2: Shuffle de datos
        shuffled_file = self.data_cleaner.shuffle_data(clean_data_file, self.shuffled_dir)
        
        # Paso 3: Estratificación
        train_file, val_file = self.data_cleaner.stratify_data(shuffled_file, self.stratified_dir)
        
        # Paso 4: Tokenización y creación de WebDataset
        train_prefix, val_prefix = self.tokenizer_processor.tokenize_and_create_webdataset(
            train_file, val_file, self.webdataset_dir)
        
        # Pre-verificación de archivos para asegurar que todos sean válidos
        self._verify_shards(self.webdataset_dir)
        
        # Paso 5: Crear pipelines de datos con WebDataset
        self.train_dataset = self.tokenizer_processor.create_webdataset_datapipeline(
            train_prefix, self.shuffle_buffer_size, is_train=True)
        self.val_dataset = self.tokenizer_processor.create_webdataset_datapipeline(
            val_prefix, self.shuffle_buffer_size, is_train=False)
        
        print("Configuración de dataset completada.")
    
    def _verify_shards(self, webdataset_dir: Path):
        """Verifica que todos los shards sean archivos TAR válidos."""
        print("Verificando integridad de shards...")
        shard_files = list(webdataset_dir.glob("*.tar"))
        for i, shard in enumerate(shard_files):
            if i % 10 == 0:  # Verificar solo 1 de cada 10 para rapidez
                try:
                    with tarfile.open(shard, "r") as tar:
                        # Simplemente listar los primeros 5 miembros es suficiente para verificar
                        for j, member in enumerate(tar):
                            if j >= 5:
                                break
                except Exception as e:
                    print(f"Error en shard {shard}: {e}")
                    # Si se encontrara un shard corrupto, podríamos eliminarlo o regenerarlo
                    # Por ahora solo reportamos
        print(f"Verificación completada. Encontrados {len(shard_files)} archivos de shards.")

    def train_dataloader(self) -> DataLoader:
        """Dataloader para entrenamiento."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3,
            drop_last=True,
            multiprocessing_context='spawn'  # Cambiado de 'fork' a 'spawn'
        )

    def val_dataloader(self) -> DataLoader:
        """Dataloader para validación."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized. Call setup() first.")
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            multiprocessing_context='spawn'  # Cambiado de 'fork' a 'spawn'
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
        self.model = self.model.to(torch.bfloat16) 
        self.model.tokenizer = tokenizer
        self.model = torch.compile(self.model, backend="inductor",
                                   mode="max-autotune", fullgraph=True)
        self.lr = lr
        self.next_token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.validation_step_outputs = []
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        text = batch["text"].to(self.device)
        logits = self.model(text)
        next_token_logits = logits[:, :-1]
        next_token_targets = text[:, 1:]
        
        # Cálculo de la pérdida
        loss = self.next_token_loss_fn(
            next_token_logits.reshape(-1, self.vocab_size),
            next_token_targets.reshape(-1)
        )
        
        # Añadir cálculo de accuracy
        predicted_tokens = torch.argmax(next_token_logits, dim=-1)
        mask = next_token_targets != self.pad_token_id  # Ignorar tokens de padding
        correct = (predicted_tokens == next_token_targets) & mask
        total_valid = mask.sum().float()
        if total_valid > 0:  # Prevenir división por cero
            accuracy = correct.sum().float() / total_valid
            # Añadir batch_size explícitamente
            self.log("train_accuracy", accuracy, prog_bar=True, batch_size=text.size(0))
        
        # Añadir batch_size explícitamente
        self.log("train_loss", loss, prog_bar=True, batch_size=text.size(0))
        return loss
    def on_validation_epoch_start(self):
        # Reiniciar acumuladores al inicio de cada época
        self.validation_step_outputs = []

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        text = batch["text"].to(self.device)
        # Extraer dominios del batch
        domains = [str(d) for d in batch.get("domain", ["unknown"] * text.size(0))]
        
        logits = self.model(text)
        next_token_logits = logits[:, :-1]
        next_token_targets = text[:, 1:]
        
        # Cálculo de la pérdida
        loss = self.next_token_loss_fn(
            next_token_logits.reshape(-1, self.vocab_size),
            next_token_targets.reshape(-1)
        )
        
        # Cálculo de accuracy global (mantener código existente)
        predicted_tokens = torch.argmax(next_token_logits, dim=-1)
        mask = next_token_targets != self.pad_token_id  # Ignorar tokens de padding
        correct = (predicted_tokens == next_token_targets) & mask
        total_valid = mask.sum().float()
        if total_valid > 0:
            accuracy = correct.sum().float() / total_valid
            self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True, batch_size=text.size(0))
        
        # Guardar resultados por ejemplo para procesamiento posterior
        example_results = []
        for i in range(text.size(0)):
            example_correct = correct[i].sum().item()
            example_total = mask[i].sum().item()
            example_domain = domains[i]
            
            example_results.append({
                "domain": example_domain,
                "correct": example_correct,
                "total": example_total
            })
        
        self.validation_step_outputs.extend(example_results)
        self.log("validation_loss", loss, prog_bar=True, sync_dist=True, batch_size=text.size(0))
        return loss
    def on_validation_epoch_end(self):
        # Recopilar resultados de todos los procesos si estamos en entrenamiento distribuido
        if self.trainer.world_size > 1:
            all_results = self.all_gather(self.validation_step_outputs)
            # Aplanar los resultados
            gathered_results = []
            for results_list in all_results:
                gathered_results.extend(results_list)
            self.validation_step_outputs = gathered_results
        
        # Organizar resultados por dominio
        domain_stats = {}
        
        for result in self.validation_step_outputs:
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}
            
            domain_stats[domain]["correct"] += result["correct"]
            domain_stats[domain]["total"] += result["total"]
        
        # Calcular y registrar accuracy por dominio
        for domain, stats in domain_stats.items():
            if stats["total"] > 0:
                domain_accuracy = stats["correct"] / stats["total"]
                # Añadir batch_size=1 ya que estos son metrics de época completa
                self.log(f"val_accuracy_{domain}", domain_accuracy, sync_dist=True, batch_size=1)
        
        # Calcular diferencia entre dominios principales
        domains = list(domain_stats.keys())
        if len(domains) >= 2:
            main_domains = [d for d in domains if d in ["fineweb", "math"]]
            if len(main_domains) >= 2 and all(domain_stats[d]["total"] > 0 for d in main_domains):
                acc_diff = abs(
                    domain_stats[main_domains[0]]["correct"] / domain_stats[main_domains[0]]["total"] -
                    domain_stats[main_domains[1]]["correct"] / domain_stats[main_domains[1]]["total"]
                )
                # Añadir batch_size=1 para métricas de época
                self.log("val_accuracy_domain_diff", acc_diff, sync_dist=True, batch_size=1)
        
        # Limpiar acumuladores
        self.validation_step_outputs = []
        
    def configure_optimizers(self):
        """
        Configura el optimizador Muon para el modelo.
        
        - Identifica parámetros matriciales para Muon (matrices 2D)
        - Los parámetros no matriciales y embeddings/LM heads se optimizan con AdamW
        - Asegura weight decay uniforme para todos los parámetros, incluyendo RMSNorm
        
        Returns:
            dict: Configuración de optimizador y scheduler
        """
        model = self.model
        
        # Separar parámetros matriciales (para Muon) de no matriciales (para AdamW)
        muon_params = [
            p for name, p in model.named_parameters() 
            if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name and p.requires_grad
        ]
        
        adamw_params = [
            p for name, p in model.named_parameters() 
            if (p.ndim != 2 or "embed_tokens" in name or "lm_head" in name) and p.requires_grad
        ]
        
        # Configurar el optimizador Muon con los parámetros separados
        optimizer = Muon(
            lr=self.lr,
            wd=0.013,  # Weight decay aplicado uniformemente, incluyendo gamma RMSNorm
            muon_params=muon_params,
            adamw_params=adamw_params,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_betas=(0.9, 0.99),
            adamw_eps=1e-8
        )
        
        # Programar el learning rate usando el mismo scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=64500,
            T_mult=1,
            eta_min=self.lr / 7
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
# --- Función de Utilidad para Inspeccionar Parámetros (la misma de antes) ---
def inspect_model_parameters(model: torch.nn.Module, model_name: str = "Model"):
    print(f"\n--- Inspecting Parameters for {model_name} ---")
    total_params = 0
    dtype_counts = {}

    for name, param in model.named_parameters():
        if param.requires_grad: # Solo parámetros entrenables
            num_params = param.numel()
            total_params += num_params
            param_dtype = param.dtype

            print(f"  Parameter: {name:<70} | Shape: {str(param.shape):<25} | Dtype: {str(param_dtype):<15} | Num params: {num_params}")

            if param_dtype not in dtype_counts:
                dtype_counts[param_dtype] = 0
            dtype_counts[param_dtype] += num_params

    print(f"\n  Total trainable parameters: {total_params}")
    print(f"  Parameter Dtype Distribution:")
    for dtype, count in dtype_counts.items():
        percentage = (count / total_params) * 100 if total_params > 0 else 0
        print(f"    - {str(dtype):<15}: {count} params ({percentage:.2f}%)")

    # También podemos inspeccionar los buffers
    buffer_dtype_counts = {}
    total_buffer_elements = 0
    print(f"\n  Buffer Dtype Distribution:")
    for name, buffer_tensor in model.named_buffers():
        num_elements = buffer_tensor.numel()
        total_buffer_elements += num_elements
        buffer_dtype = buffer_tensor.dtype
        print(f"  Buffer:    {name:<70} | Shape: {str(buffer_tensor.shape):<25} | Dtype: {str(buffer_dtype):<15} | Num elements: {num_elements}")
        if buffer_dtype not in buffer_dtype_counts:
            buffer_dtype_counts[buffer_dtype] = 0
        buffer_dtype_counts[buffer_dtype] += num_elements

    if total_buffer_elements > 0:
        for dtype, count in buffer_dtype_counts.items():
            percentage = (count / total_buffer_elements) * 100
            print(f"    - {str(dtype):<15}: {count} elements ({percentage:.2f}%)")
    else:
        print("    No buffers found or all buffers are empty.")

    print(f"--- End of Inspection for {model_name} ---\n")
############################################
# BLOQUE PRINCIPAL: TRAINING E INFERENCIA
############################################
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method(method='spawn', force=True)
    
    import argparse
    import sys
    import os
    import torch
    from transformers import AutoTokenizer
    import pytorch_lightning as pl


    parser = argparse.ArgumentParser(
        description="Entrenamiento o Inferencia del modelo FANformer (preentrenamiento)"
    )
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--epochs_text", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=200, help="Longitud máxima a generar (tokens adicionales)")
    parser.add_argument("--min_length", type=int, default=1, help="Longitud mínima a generar antes de considerar EOS")
    parser.add_argument("--top_k", type=int, default=100, help="Valor de top_k para sampling")
    parser.add_argument("--top_p", type=float, default=0.85, help="Valor de top_p para sampling")
    parser.add_argument("--temperature", type=float, default=0.9, help="Factor de temperatura para escalado de logits")
    parser.add_argument("--p", type=float, default=0.14, help="Proporción de modelado periódico para FANformer")
    parser.add_argument("--stride", type=int, default=16, help="Solapamiento entre fragmentos de texto")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    
    config = {
        "VOCAB_SIZE": tokenizer.vocab_size,
        "EMBED_DIM": 768,
        "NUM_HEADS": 12,
        "NUM_DECODER_LAYERS": 12,
        "FF_DIM": 2048, 
        "LR": 6e-4,
        "batch_size": args.batch_size,
        "NUM_GQA_GROUPS": 6,
        "BASE_DROPOUT": 0.08,
        "p": args.p
    }
    mode_config = {
        "MAX_SEQ_LEN": 1024,
        "max_examples": 300000
    }
    torch.cuda.empty_cache()

    if args.mode == "train":
        print(f"[INFO] Entrenamiento en modo DECODER-ONLY utilizando FineWeb-Edu (sample-10BT)")
        print(f"[CONFIG] Utilizando FANformer con p={args.p}")
        print(f"[CONFIG] Longitud máxima de secuencia: {mode_config['MAX_SEQ_LEN']}")
        print(f"[CONFIG] Tamaño de batch: {config['batch_size']}")
        print(f"[CONFIG] Stride para fragmentos: {args.stride}")
    
        # Inicializar el data module
        data_module = FinewebTextDataModule(
            tokenizer=tokenizer, 
            max_seq_len=mode_config["MAX_SEQ_LEN"],
            batch_size=config["batch_size"],
            max_examples=mode_config["max_examples"],
            stride=args.stride
        )

        # Inicializar el modelo
        model_module = MultiModalLightningModule(
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
        # << --- INSPECCIÓN ANTES DE LA CONVERSIÓN (OPCIONAL) --- >>
        if hasattr(model_module, 'model') and isinstance(model_module.model, torch.nn.Module):
            inspect_model_parameters(model_module.model, "MultiModalModel (interno, ANTES de .to(bfloat16))")
        else:
            inspect_model_parameters(model_module, "MultiModalLightningModule (ANTES de .to(bfloat16))")

        trainer_config = {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "precision": "bf16-true",
            "accumulate_grad_batches": 1,
            "enable_checkpointing": False,
            "val_check_interval": 150,
            "limit_val_batches": 15,
            "gradient_clip_val": 1,
            "gradient_clip_algorithm": "norm",
            "strategy": "ddp_find_unused_parameters_true",
            "enable_progress_bar": True
            # Eliminado: "max_steps": total_steps
        }
        trainer = pl.Trainer(max_epochs=args.epochs_text, **trainer_config)
        trainer.fit(model_module, datamodule=data_module)
        
        # Guardar el modelo
        os.makedirs('./modelo_final', exist_ok=True)
        final_ckpt = f"./modelo_final/modelo_final_fanformer1.ckpt"
        trainer.save_checkpoint(final_ckpt)
        print(f"[SUCCESS] Modelo guardado exitosamente en {final_ckpt}")
    else:
        # Inferencia
        checkpoint_path = f"./modelo_final/fanformer_finetuned_final.ckpt"
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] No se encontró el checkpoint en {checkpoint_path}")
            checkpoint_path = "./modelo_final/fanformer_finetuned_final.ckpt"
            print(f"[INFO] Intentando con modelo genérico: {checkpoint_path}")
            
        print(f"[INFO] Modo de inferencia activado utilizando FANformer con p={args.p}")
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
        
        print("\n=== Modo Inferencia ===")
        print("Ingresa tus prompts o escribe 'salir' para terminar")
        print("="*50)
        
        while True:
            prompt = input("\nPrompt> ")
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
            print(f"Texto generado: {generated_text}")
            print("="*50)
            
        print("\n¡Gracias por usar el modelo de inferencia!")
