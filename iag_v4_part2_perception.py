"""
IAG Modular V4 - Parte 2
Percepción multimodal y codificación unificada.

Se conecta con la Parte 1 mediante vectores NumPy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Union
import math

from hca_layer import HybridCompressiveAttention, HCAConfig, precompute_rope_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Dispositivo y Tipos Globales
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PerceptionConfig:
    input_dim: int = 256
    n_heads: int = 128
    latent_dim: int = 7168
    cnn_channels: int = 16
    rnn_hidden: int = 128


# ============================================================
# Procesador Espacial (CNN)
# ============================================================

class CNNSpatialProcessor(nn.Module):
    def __init__(self, channels: int = 16, in_channels: int = 3, device: torch.device = DEVICE):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self._target_device = device
        # Pesos en formato (out_channels, in_channels, kH, kW) para F.conv2d
        kernels = (torch.randn(channels, in_channels, 3, 3, dtype=DTYPE) * math.sqrt(2.0 / (3 * 3 * in_channels))).to(device)
        self.register_buffer("kernels", kernels)

    def forward(self, image: Union[torch.Tensor, Any]) -> torch.Tensor:
        target_device = self.kernels.device
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=DTYPE, device=target_device)
        else:
            image = image.to(device=target_device, dtype=DTYPE)
        
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)   # (H,W) -> (1,1,H,W)
        elif image.ndim == 3:
            if image.shape[-1] in (1, 3, 4):
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
        
        if image.shape[1] == 1 and self.in_channels == 3:
            image = image.repeat(1, 3, 1, 1)
    
        out = F.conv2d(image, self.kernels, padding=1)
        out = torch.tanh(out)
        return torch.mean(out, dim=(-2, -1))


# ============================================================
# Procesador Temporal (RNN)
# ============================================================

class RNNTemporalProcessor(nn.Module):
    def __init__(self, input_dim: int = 256, hidden: int = 32, device: torch.device = DEVICE):
        super().__init__()
        self.hidden = hidden

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden, batch_first=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        target_dev = self.device
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence, dtype=DTYPE, device=target_dev)
        else:
            sequence = sequence.to(device=target_dev, dtype=DTYPE)

        if sequence.ndim == 1:
            sequence = sequence.view(1, 1, -1)
        elif sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)

        _, h_n = self.rnn(sequence)
        return h_n[-1]


# ============================================================
# Codificador Unificado
# ============================================================

class SparseHDCProjection(nn.Module):
    def __init__(self, in_features: int = 49152, out_features: int = 1024, sparsity: float = 0.05, device: torch.device = DEVICE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Generar conectividad dispersa
        nnz = int(in_features * out_features * sparsity)
        indices = torch.stack([
            torch.randint(0, out_features, (nnz,)),
            torch.randint(0, in_features, (nnz,))
        ])
        values = torch.randn(nnz) * math.sqrt(2.0 / (in_features + out_features))
        
        self.register_buffer("sparse_W", torch.sparse_coo_tensor(indices, values, (out_features, in_features)).coalesce())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_device = self.device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=DTYPE, device=target_device)
        else:
            x = x.to(device=target_device, dtype=DTYPE)

        if x.ndim == 1:
            return torch.sparse.mm(self.sparse_W, x.unsqueeze(1)).squeeze(1)
        # Para entradas 2D (B, in_features): (W_sparse @ X.T).T -> (B, out_features)
        return torch.sparse.mm(self.sparse_W, x.t()).t()

class UnifiedEncoder(nn.Module):
    """
    Convierte cualquier entrada en representación cognitiva común (Tensor Latente).
    """
    def __init__(self, cfg: PerceptionConfig | None = None, device: torch.device = DEVICE):
        super().__init__()
        self.cfg = cfg or PerceptionConfig()
        self.device = device

        self.cnn = CNNSpatialProcessor(self.cfg.cnn_channels, device=self.device)
        self.rnn = RNNTemporalProcessor(self.cfg.input_dim, self.cfg.rnn_hidden, device=self.device)

        self.text_embedding = nn.Embedding(
            num_embeddings=256, 
            embedding_dim=self.cfg.latent_dim, 
            device=self.device
        )
        # Inicialización estándar para embeddings
        nn.init.normal_(self.text_embedding.weight, mean=0, std=0.02)

        proj_img_tensor = torch.randn(
            self.cfg.latent_dim, self.cfg.cnn_channels, dtype=DTYPE
        ) * math.sqrt(2.0 / (self.cfg.cnn_channels + self.cfg.latent_dim))
        self.proj_img = nn.Parameter(proj_img_tensor.to(self.device))

        proj_seq_tensor = torch.randn(
            self.cfg.latent_dim, self.cfg.rnn_hidden, dtype=DTYPE
        ) * math.sqrt(2.0 / (self.cfg.rnn_hidden + self.cfg.latent_dim))
        self.proj_seq = nn.Parameter(proj_seq_tensor.to(self.device))

    def encode_text(self, text: str) -> torch.Tensor:
        bytes_data = text.encode("utf-8")
        vec = torch.tensor(list(bytes_data), dtype=torch.long, device=self.device)

        if vec.numel() < self.cfg.input_dim:
            vec = F.pad(vec, (0, self.cfg.input_dim - vec.numel()), value=0)
        else:
            vec = vec[:self.cfg.input_dim]

        embedded_tokens = self.text_embedding(vec)

        return embedded_tokens

    def encode_image(self, image: Any) -> torch.Tensor:
        f = self.cnn.forward(image)
        # F.linear soporta adecuadamente tensores 1D (C,) y 2D (B, C)
        return torch.tanh(F.linear(f, self.proj_img))

    def encode_sequence(self, sequence: Any) -> torch.Tensor:
        h = self.rnn.forward(sequence)
        return torch.tanh(F.linear(h, self.proj_seq))


# ============================================================
# Hub de Percepción
# ============================================================

class PerceptionHub(nn.Module):
    """
    Entrada unificada para la canalización cognitiva.
    """
    def __init__(self, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.cfg = PerceptionConfig()
        self.encoder = UnifiedEncoder(self.cfg, device=self.device)

        # 1. Configurar los parámetros de HCA a partir de PerceptionConfig
        hca_config = HCAConfig(
            d_model=7168,
            n_heads=128,
            d_head=128,
            d_rope=64,
            r_kv=512,
            r_q=1536,
            max_seq_len=131072
        )
        
        # 2. Instanciar la capa HCA
        self.hca_attention = HybridCompressiveAttention(hca_config).to(self.device)

        # 3. Precomputar los caché de frecuencias cos/sin de RoPE
        cos_rope, sin_rope = precompute_rope_cache(
            seq_len=131072,
            dim=64,
            theta=10000.0,
            device=self.device,
            dtype=DTYPE
        )
        self.register_buffer("cos_rope", cos_rope, persistent=False)
        self.register_buffer("sin_rope", sin_rope, persistent=False)

    def perceive_text(self, text: str, is_causal: bool = True) -> torch.Tensor:
        # Generar tokens embebidos: (seq_len, latent_dim)
        tokens = self.encoder.encode_text(text).unsqueeze(0)  # (1, seq_len, latent_dim)
        
        seq_len = tokens.size(1)
        
        # Extraer las tablas RoPE para la longitud actual de la secuencia
        cos = self.cos_rope[:seq_len]
        sin = self.sin_rope[:seq_len]

        # Pasar el tensor por la atención comprimida HCA
        latent = self.hca_attention(
            h=tokens,
            cos_rope=cos,
            sin_rope=sin,
            causal=is_causal
        )  # (1, seq_len, latent_dim)

        # Reducción de secuencia y normalización L2 final
        latent_vec = latent.mean(dim=1).squeeze(0)  # (latent_dim,)
        return F.normalize(latent_vec, p=2.0, dim=-1, eps=1e-8)

    def perceive_image(self, image: Any) -> torch.Tensor:
        return self.encoder.encode_image(image)

    def perceive_sequence(self, sequence: Any) -> torch.Tensor:
        return self.encoder.encode_sequence(sequence)


if __name__ == "__main__":
    hub = PerceptionHub()
    v = hub.perceive_text("Hola mundo desde IAG V4")

    print(f"Percepción ejecutada en: {hub.device.type.upper()}")
    print("Tensor latente resultante (shape):", v.shape)
    print("Tipo de tensor:", v.dtype)