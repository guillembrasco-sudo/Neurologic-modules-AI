import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DTYPE


@dataclass
class HCAConfig:
    """Hiperparámetros del módulo Hybrid Compressive Attention (HCA)."""
    d_model: int = 5120          # dimensión del modelo (n_heads * d_head)
    n_heads: int = 40            # número de cabezas de atención
    d_head: int = 128            # dimensión de contenido por cabeza
    d_rope: int = 64             # dimensión del sub-canal RoPE desacoplado (compartido)
    r_kv: int = 512              # rango latente comprimido de K/V (<< n_heads * d_head)
    r_q: int = 1536              # rango latente comprimido de Q (solo afecta activaciones)
    rope_theta: float = 10000.0
    max_seq_len: int = 131072
    rmsnorm_eps: float = 1e-6
    dropout: float = 0.0


class RMSNorm(nn.Module):
    """Normalización RMS aplicada a los vectores latentes comprimidos.

    Se aplica sobre c_t^{KV} y c_t^{Q} para evitar drift de escala tras la
    compresión de bajo rango, que es una fuente conocida de inestabilidad
    numérica en BF16/FP16 (ver Sección 5).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cálculo en fp32 para evitar underflow/overflow del cuadrado en fp16.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return (x_normed.type_as(x)) * self.weight


def precompute_rope_cache(seq_len: int, dim: int, theta: float, device, dtype):
    """Precomputa cos/sin de RoPE para el sub-canal posicional desacoplado."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica RoPE únicamente al sub-canal posicional desacoplado (dim = d_rope)."""
    return (x * cos) + (rotate_half(x) * sin)


class HybridCompressiveAttention(nn.Module):
    """Hybrid Compressive Attention (HCA).

    Fusiona:
      - Compresión latente de bajo rango de K/V compartida entre cabezas (estilo MLA).
      - RoPE desacoplado en un sub-canal posicional compartido (estilo MQA para RoPE).
      - Absorción de matrices (weight absorption) para inferencia sin reconstruir K/V.
      - Ejecución IO-aware vía scaled_dot_product_attention (backend flash/mem-efficient).

    KV Cache cacheado: únicamente `c_kv` (dim r_kv) y `k_rope` (dim d_rope) por token,
    en lugar de K y V completos por cabeza.
    """

    def __init__(self, config: HCAConfig):
        super().__init__()
        self.cfg = config
        d_model, n_heads, d_head = config.d_model, config.n_heads, config.d_head
        d_rope, r_kv, r_q = config.d_rope, config.r_kv, config.r_q

        # --- Down-projections (compresión a espacio latente) ---
        self.W_DKV = nn.Linear(d_model, r_kv, bias=False)          # h_t -> c_t^{KV}
        self.W_DQ = nn.Linear(d_model, r_q, bias=False)            # h_t -> c_t^{Q}
        self.kv_norm = RMSNorm(r_kv, eps=config.rmsnorm_eps)
        self.q_norm = RMSNorm(r_q, eps=config.rmsnorm_eps)

        # --- Up-projections (reconstrucción de contenido, por cabeza) ---
        self.W_UK = nn.Linear(r_kv, n_heads * d_head, bias=False)   # c^{KV} -> K^C (todas las cabezas)
        self.W_UV = nn.Linear(r_kv, n_heads * d_head, bias=False)   # c^{KV} -> V^C (todas las cabezas)
        self.W_UQ = nn.Linear(r_q, n_heads * d_head, bias=False)    # c^{Q}  -> Q^C (todas las cabezas)

        # --- Sub-canal RoPE desacoplado ---
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)          # h_t -> k_t^{R} (compartido, 1 sola "cabeza")
        self.W_QR = nn.Linear(r_q, n_heads * d_rope, bias=False)    # c_t^{Q} -> q_{t,i}^{R} (por cabeza)

        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.dropout = config.dropout
        self.scale = 1.0 / math.sqrt(d_head + d_rope)

        # Buffer para la absorción de matrices (se construye en modo inferencia)
        self._absorbed_ready = False
        self.register_buffer("W_Q_absorbed", None, persistent=False)  # [n_heads, r_q, r_kv]

    # ------------------------------------------------------------------
    # Absorción de matrices: W_Q_absorbed[i] = W_UQ_i^T @ W_UK_i
    # Permite calcular q^T k directamente en el espacio latente (r_q x r_kv)
    # sin reconstruir K^C. Se recomienda llamar tras cargar pesos entrenados
    # y antes de servir en inferencia (equivalente a "kernel fusion" a nivel
    # de pesos).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_for_inference(self):
        n_heads, d_head, r_q, r_kv = (
            self.cfg.n_heads, self.cfg.d_head, self.cfg.r_q, self.cfg.r_kv
        )
        W_UQ = self.W_UQ.weight.view(n_heads, d_head, r_q)   # [n_heads, d_head, r_q]
        W_UK = self.W_UK.weight.view(n_heads, d_head, r_kv)  # [n_heads, d_head, r_kv]
        # W_Q_absorbed[i] = W_UQ_i^T @ W_UK_i  -> [r_q, r_kv] por cabeza
        absorbed = torch.einsum("hdq,hdk->hqk", W_UQ, W_UK)
        self.W_Q_absorbed = absorbed  # [n_heads, r_q, r_kv]
        self._absorbed_ready = True

    def forward(
        self,
        h: torch.Tensor,
        cos_rope: torch.Tensor,
        sin_rope: torch.Tensor,
        kv_cache: Optional["HCAKVCache"] = None,
        cache_position: Optional[int] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model] estados ocultos de entrada.
            cos_rope, sin_rope: tablas RoPE precomputadas para las posiciones actuales.
            kv_cache: caché comprimido opcional (para decode autoregresivo).
            cache_position: posición inicial de este bloque dentro de la secuencia cacheada.
            causal: si aplicar máscara causal (True en entrenamiento/prefill).

        Returns:
            [batch, seq_len, d_model]
        """
        B, T, _ = h.shape
        n_heads, d_head, d_rope = self.cfg.n_heads, self.cfg.d_head, self.cfg.d_rope

        # --- 1. Compresión latente (down-projection) + normalización ---
        c_kv = self.kv_norm(self.W_DKV(h))   # [B, T, r_kv]
        c_q = self.q_norm(self.W_DQ(h))      # [B, T, r_q]

        # --- 2. Sub-canal RoPE desacoplado ---
        k_rope = self.W_KR(h)                                    # [B, T, d_rope]  (compartido entre cabezas)
        q_rope = self.W_QR(c_q).view(B, T, n_heads, d_rope)       # [B, T, H, d_rope]

        k_rope_rot = apply_rope(k_rope, cos_rope, sin_rope)                       # [B, T, d_rope]
        q_rope_rot = apply_rope(q_rope, cos_rope[:, None, :], sin_rope[:, None, :])  # [B, T, H, d_rope]

        # --- 3. Actualización del KV Cache comprimido (solo c_kv y k_rope) ---
        if kv_cache is not None:
            c_kv, k_rope_rot = kv_cache.update(c_kv, k_rope_rot, cache_position)
            S = c_kv.shape[1]  # longitud total en cache (histórico + actual)
            scores_content = torch.einsum("bthq,hqk,bsk->bhts", q_rope_rot, self.W_Q_absorbed, c_kv)
        else:
            S = T
            k_content = self.W_UK(c_kv).view(B, S, n_heads, d_head)

        # --- 4. Reconstrucción de Q (contenido) por cabeza ---
        q_content = self.W_UQ(c_q).view(B, T, n_heads, d_head)          # [B, T, H, d_head]
        q = torch.cat([q_content, q_rope_rot], dim=-1)                  # [B, T, H, d_head+d_rope]

        # --- 5. Reconstrucción de K, V (contenido) por cabeza ---
        #     NOTA: en un kernel de producción con absorción activada (prepare_for_inference),
        #     este paso se sustituye por el cómputo directo q @ W_Q_absorbed @ c_kv^T,
        #     evitando materializar k_content por completo (ver Sección 1.5).
        k_content = self.W_UK(c_kv).view(B, S, n_heads, d_head)         # [B, S, H, d_head]
        v_content = self.W_UV(c_kv).view(B, S, n_heads, d_head)         # [B, S, H, d_head]
        k_rope_b = k_rope_rot.unsqueeze(2).expand(-1, -1, n_heads, -1)  # broadcast a todas las cabezas
        k = torch.cat([k_content, k_rope_b], dim=-1)                    # [B, S, H, d_head+d_rope]
        v = v_content                                                   # [B, S, H, d_head]

        # --- 6. Atención IO-aware (tiling/online-softmax vía SDPA backend flash) ---
        q_ = q.transpose(1, 2)  # [B, H, T, d_head+d_rope]
        k_ = k.transpose(1, 2)  # [B, H, S, d_head+d_rope]
        v_ = v.transpose(1, 2)  # [B, H, S, d_head]

        attn_out = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal and (T == S),  # solo causal real en prefill; en decode T=1 no aplica
            scale=self.scale,
        )  # [B, H, T, d_head]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, n_heads * d_head)
        return self.o_proj(attn_out)


class HCAKVCache:
    """KV Cache comprimido: almacena únicamente c_t^{KV} (r_kv) y k_t^{R} (d_rope)
    por token, en lugar de K/V completos por cabeza.

    Soporta cuantización opcional a FP8/INT4 para el almacenamiento persistente
    (Sección 2.2); aquí se muestra la variante en punto flotante para claridad.
    """

    def __init__(self, batch_size: int, max_seq_len: int, r_kv: int, d_rope: int,
                 device, dtype=DTYPE):
        self.c_kv = torch.zeros(batch_size, max_seq_len, r_kv, device=device, dtype=dtype)
        self.k_rope = torch.zeros(batch_size, max_seq_len, d_rope, device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, c_kv_new: torch.Tensor, k_rope_new: torch.Tensor,
               position: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_new, _ = c_kv_new.shape
        pos = position if position is not None else self.seq_len
        self.c_kv[:, pos:pos + T_new] = c_kv_new
        self.k_rope[:, pos:pos + T_new] = k_rope_new
        self.seq_len = pos + T_new
        return self.c_kv[:, :self.seq_len], self.k_rope[:, :self.seq_len]

    def memory_bytes(self, dtype_bytes: int = 2) -> int:
        """Footprint de memoria actual del cache, en bytes."""
        return (self.c_kv[:, :self.seq_len].numel()
                + self.k_rope[:, :self.seq_len].numel()) * dtype_bytes


# Ejemplo de uso e inspección de dimensiones:
config_hca_70b = HCAConfig(
    d_model=7168,
    n_heads=128,       # 128 cabezas de atención
    d_head=128,
    d_rope=64,
    r_kv=512,          # Compresión KV (~14x respecto a d_model)
    r_q=1536,
    max_seq_len=131072
)