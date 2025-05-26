# model_architecture.py
# Arquitectura del modelo FANformer con todas las innovaciones

import math
import random
from typing import Optional, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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
# GATED RESIDUALS - HYPERCONNECTIONS
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

        # Capas de normalización para la entrada (Pre-Norm en primer bloque o QKV-Norm para los demás)
        self.norm = nn.RMSNorm(embed_dim, eps=1e-5)

        # Capas de dropout (simplificadas)
        self.attention_dropout = progressive_dropout(dropout, depth=layer_index) # Usar layer_index
        self.output_dropout = progressive_dropout(dropout, depth=layer_index) # Usar layer_index

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

        # Mantener la lógica original de manejo de máscara opcional
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

        try:
            # Usar flash attention sin padding, pasando q_norm
            output_unpadded = self.flash_attn_varlen_func(
                q_norm, k_norm, v_unpadded,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=self.attention_dropout.p,
                softmax_scale=None,
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

        # Preparar tensores para Flash Attention (requiere shape [B, S, H, D])
        q_trans = q_norm.permute(0, 2, 1, 3)
        k_trans = k_norm.permute(0, 2, 1, 3)
        v_trans = v.permute(0, 2, 1, 3)

        # Verificación de dimensiones
        if q_trans.size(-1) != k_trans.size(-1):
            raise ValueError(f"Las dimensiones de head no coinciden: q={q_trans.size(-1)}, k={k_trans.size(-1)}")

        try:
            # Aplicar Flash Attention
            output = self.flash_attn_func(
                q_trans, k_trans, v_trans,
                dropout_p=self.attention_dropout.p,
                softmax_scale=None,
                causal=is_causal
            )

            if output is None:
                raise ValueError("flash_attn_func devolvió None. Verifica las dimensiones y tipos de los tensores de entrada.")

            # Volver a la forma original
            output = output.permute(0, 2, 1, 3)
            return output

        except Exception as e:
            raise RuntimeError(f"Error en flash_attn_func: {e}")

    def forward(self, X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, causal: bool = True) -> torch.Tensor:
        B, T, _ = X.shape
        norm_func = self.norm

        # Implementación de HybridNorm*
        if self.use_pre_norm:
            # Primer bloque: Pre-Norm en atención
            X_norm = norm_func(X)
            # Proyecciones para Q, K, V con FANformer
            Q = self.Wq(X_norm)  # [B, T, num_heads, head_dim]
            K = self.Wk(X_norm)  # [B, T, num_heads, head_dim]
            V = self.Wv(X_norm)  # [B, T, num_heads, head_dim]
        else:
            # Otros bloques: QKV-Norm
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

        # Reorganizar la salida y aplicar proyección final
        out = attn_output.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, T, self.embed_dim)
        out = self.output_dropout(self.out_proj(out))

        return out

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
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature

        # 2. Aplicar Top-K
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, V))
            kth_value = values[:, -1].unsqueeze(1)
            logits[logits < kth_value] = -float('inf')

        # 3. Aplicar Top-P (Nucleus Sampling)
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')

        # 4. Muestrear desde la distribución modificada
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def forward_inference(self, prompt_tokens: torch.Tensor, max_length: int = 100, min_length: int = 1,
                          top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """
        Genera texto usando muestreo (temperatura, top-k, top-p) sin beam search ni penalización.
        """
        if self.tokenizer is None:
            raise ValueError("El tokenizador no ha sido asignado al modelo. Asigna 'model.tokenizer = tokenizer'.")

        generated = prompt_tokens
        cur_length = generated.size(1)

        # Bucle de generación
        for _ in range(max_length):
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
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_token_id is not None and cur_length > min_length and (next_token == eos_token_id).all():
                 break

        return generated
