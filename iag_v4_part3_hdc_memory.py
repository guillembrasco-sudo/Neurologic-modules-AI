"""
IAG Modular V4 - Parte 3
Memoria Hiperdimensional (HDC) como bus cognitivo.

Incluye:
- Generación de hiper-vectores binarios y bipolares
- Binding / bundling / permutation
- Memoria asociativa eficiente
- One-shot learning
- Consolidación episódica y semántica
- Limpieza de memoria por relevancia
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ============================================================
# Dispositivo y Tipos Globales
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_bipolar(x: torch.Tensor) -> torch.Tensor:
    # torch.where acepta valores escalares nativos sin necesidad de envolverlos en torch.tensor()
    return torch.where(x >= 0.0, 1.0, -1.0).to(dtype=DTYPE)


def _roll(vec: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(vec, shifts=int(shift), dims=0)


def _stable_hash(key: str) -> int:
    # Hash FNV-1a de 32 bits para asegurar consistencia multiplataforma
    h = 2166136261
    for ch in key.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


# ============================================================
# Fábrica de Hipervectores
# ============================================================

class HypervectorFactory(nn.Module):
    def __init__(self, dim: int = 10000, seed: Optional[int] = 7, device: torch.device = DEVICE):
        super().__init__()
        self.dim = int(dim)
        self.seed = seed if seed is not None else 7
        self._target_device = device
        self._cache: Dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return self._target_device

    def random_bipolar(self) -> torch.Tensor:
        r = torch.rand(self.dim, device=self.device)
        return torch.where(r >= 0.5, 1.0, -1.0).to(dtype=DTYPE)

    def random_binary(self) -> torch.Tensor:
        return torch.randint(0, 2, size=(self.dim,), dtype=DTYPE, device=self.device)

    def basis(self, key: str) -> torch.Tensor:
        if key in self._cache:
            return self._cache[key].to(device=self.device)

        key_hash = _stable_hash(key)
        unique_seed = (self.seed ^ key_hash) & 0xFFFFFFFF
        gen = torch.Generator(device="cpu").manual_seed(unique_seed)
        r = torch.rand(self.dim, generator=gen)
        vec = torch.where(r >= 0.5, 1.0, -1.0).to(device=self.device, dtype=DTYPE)
        
        # Limitar tamaño de caché para evitar fuga de memoria
        if len(self._cache) < 10000:
            self._cache[key] = vec.cpu()
        return vec

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _normalize_bipolar(a.to(device=self.device, dtype=DTYPE) * b.to(device=self.device, dtype=DTYPE))

    def bundle(self, *vectors: torch.Tensor) -> torch.Tensor:
        if not vectors:
            return self.random_bipolar()
        stacked = torch.stack([v.to(device=self.device, dtype=DTYPE) for v in vectors], dim=0)
        return _normalize_bipolar(torch.sum(stacked, dim=0))

    def permute(self, vec: torch.Tensor, shift: int = 1) -> torch.Tensor:
        return _roll(vec.to(device=self.device, dtype=DTYPE), shift)

    def encode_symbol(self, symbol: str) -> torch.Tensor:
        return self.basis(f"sym::{symbol}")

    def encode_scalar(self, value: float, min_value: float = -1.0, max_value: float = 1.0) -> torch.Tensor:
        v_clamped = torch.clamp(torch.tensor((value - min_value) / (max_value - min_value + 1e-8)), 0.0, 1.0)
        v = float(v_clamped.item())
        a = self.basis("scalar_low")
        b = self.basis("scalar_high")
        return _normalize_bipolar((1.0 - v) * a + v * b)


# ============================================================
# Estructuras de Memoria HDC
# ============================================================

@dataclass
class HDCItem:
    key: str
    hypervector: torch.Tensor
    payload: Any
    weight: float = 1.0
    timestamp: float = 0.0
    tags: Tuple[str, ...] = ()
    access_count: int = 0

    def compact(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "hypervector": self.hypervector.detach().cpu().tolist() if isinstance(self.hypervector, torch.Tensor) else self.hypervector,
            "payload": self.payload,
            "weight": float(self.weight),
            "timestamp": float(self.timestamp),
            "tags": self.tags,
            "access_count": int(self.access_count),
        }


@dataclass
class RetrievalResult:
    key: str
    payload: Any
    score: float
    item: HDCItem


class HDCMemory(nn.Module):
    def __init__(self, dim: int = 10000, seed: Optional[int] = 7, capacity: int = 8192, device: torch.device = DEVICE):
        super().__init__()
        self.dim = int(dim)
        self.capacity = int(capacity)
        self._target_device = device
        self.factory = HypervectorFactory(dim=self.dim, seed=seed, device=device)

        self.register_buffer("_matrix", torch.zeros((self.capacity, self.dim), dtype=DTYPE, device=device))
        self.register_buffer("_weights", torch.zeros((self.capacity,), dtype=DTYPE, device=device))
        self.register_buffer("_timestamps", torch.zeros((self.capacity,), dtype=torch.float64, device=device))
        self.register_buffer("_access", torch.zeros((self.capacity,), dtype=torch.int32, device=device))

        self._keys: List[str] = []
        self._payloads: List[Any] = []
        self._tags: List[Tuple[str, ...]] = []

        self._size = 0
        self._cursor = 0

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return self._target_device

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        key: str,
        hypervector: torch.Tensor,
        payload: Any,
        weight: float = 1.0,
        timestamp: float = 0.0,
        tags: Tuple[str, ...] = (),
    ) -> None:
        hv = _normalize_bipolar(hypervector.to(device=self.device, dtype=DTYPE))
        item = HDCItem(
            key=key,
            hypervector=hv,
            payload=payload,
            weight=float(weight),
            timestamp=float(timestamp),
            tags=tags,
        )

        idx = self._cursor
        if self._size < self.capacity:
            self._keys.append(item.key)
            self._payloads.append(item.payload)
            self._tags.append(item.tags)
            self._size += 1
        else:
            self._keys[idx] = item.key
            self._payloads[idx] = item.payload
            self._tags[idx] = item.tags

        self._matrix[idx] = item.hypervector
        self._weights[idx] = item.weight
        self._timestamps[idx] = item.timestamp
        self._access[idx] = item.access_count
        self._cursor = (self._cursor + 1) % self.capacity

    def encode_and_store(self, key: str, payload: Any, tags: Tuple[str, ...] = ()) -> torch.Tensor:
        hv = self.factory.encode_symbol(key)
        self.add(key=key, hypervector=hv, payload=payload, tags=tags)
        return hv

    def query(self, hypervector: torch.Tensor, topk: int = 5) -> List[RetrievalResult]:
        if self._size == 0:
            return []

        q = _normalize_bipolar(hypervector.to(device=self.device, dtype=DTYPE))
        m = self._matrix[: self._size]

        sims = (m @ q) / float(self.dim)
        sims = sims + 0.0005 * self._weights[: self._size]

        k = min(topk, self._size)
        vals, idx = torch.topk(sims, k=k)

        self._access.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))

        out: List[RetrievalResult] = []
        idx_list = idx.cpu().tolist()
        vals_list = vals.cpu().tolist()

        for v, i in zip(vals_list, idx_list):
            out.append(
                RetrievalResult(
                    key=self._keys[i],
                    payload=self._payloads[i],
                    score=float(v),
                    item=HDCItem(
                        key=self._keys[i],
                        hypervector=self._matrix[i].clone(),
                        payload=self._payloads[i],
                        weight=float(self._weights[i].item()),
                        timestamp=float(self._timestamps[i].item()),
                        tags=self._tags[i],
                        access_count=int(self._access[i].item()),
                    ),
                )
            )
        return out

    def associative_recall(self, cue: torch.Tensor, threshold: float = 0.25, topk: int = 5) -> Optional[RetrievalResult]:
        results = self.query(cue, topk=topk)
        if not results:
            return None
        best = results[0]
        return best if best.score >= threshold else None

    def cleanup(self, keep_fraction: float = 0.75) -> None:
        if self._size == 0:
            return

        combined_score = self._weights[:self._size] * 0.7 + self._access[:self._size].float() * 0.3
        k = max(1, int(self._size * keep_fraction))
        _, keep = torch.topk(combined_score, k)
        if len(keep) == self._size:
            return

        new_size = len(keep)
        if new_size > 0:
            # Reasignación masiva en bloque contiguo
            self._matrix[:new_size] = self._matrix[keep]
            self._weights[:new_size] = self._weights[keep]
            self._timestamps[:new_size] = self._timestamps[keep]
            self._access[:new_size] = self._access[keep]

            keep_indices = keep.cpu().tolist()
            self._keys = [self._keys[i] for i in keep_indices]
            self._payloads = [self._payloads[i] for i in keep_indices]
            self._tags = [self._tags[i] for i in keep_indices]
        else:
            self._keys.clear()
            self._payloads.clear()
            self._tags.clear()

        # Reseteo del búfer residual
        self._matrix[new_size:] = 0.0
        self._weights[new_size:] = 0.0
        self._timestamps[new_size:] = 0.0
        self._access[new_size:] = 0

        self._size = new_size
        self._cursor = new_size % self.capacity

    def export(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "capacity": self.capacity,
            "keys": list(self._keys),
            "payloads": list(self._payloads),
            "tags": list(self._tags),
            # Se mantienen tensores puros de PyTorch para permitir guardado binario optimizado
            "matrix": self._matrix[: self._size].detach().cpu(),
            "weights": self._weights[: self._size].detach().cpu(),
            "timestamps": self._timestamps[: self._size].detach().cpu(),
            "access": self._access[: self._size].detach().cpu(),
            "size": self._size,
            "cursor": self._cursor,
        }

    def import_(self, payload: Dict[str, Any]) -> None:
        self.dim = int(payload.get("dim", self.dim))
        self.capacity = int(payload.get("capacity", self.capacity))

        self._keys = list(payload.get("keys", []))
        self._payloads = list(payload.get("payloads", []))
        self._tags = list(payload.get("tags", []))
        self._size = min(int(payload.get("size", len(self._keys))), self.capacity)
        self._cursor = int(payload.get("cursor", self._size % max(1, self.capacity)))

        self._matrix = torch.zeros((self.capacity, self.dim), dtype=DTYPE, device=self.device)
        self._weights = torch.zeros((self.capacity,), dtype=DTYPE, device=self.device)
        self._timestamps = torch.zeros((self.capacity,), dtype=torch.float64, device=self.device)
        self._access = torch.zeros((self.capacity,), dtype=torch.int32, device=self.device)

        m = payload.get("matrix", [])
        w = payload.get("weights", [])
        ts = payload.get("timestamps", [])
        ac = payload.get("access", [])

        if m:
            mt = torch.tensor(m, dtype=DTYPE, device=self.device)
            n = min(self._size, mt.shape[0])
            self._matrix[:n] = mt[:n]

        if w:
            wt = torch.tensor(w, dtype=DTYPE, device=self.device)
            n = min(self._size, wt.shape[0])
            self._weights[:n] = wt[:n]

        if ts:
            tst = torch.tensor(ts, dtype=torch.float64, device=self.device)
            n = min(self._size, tst.shape[0])
            self._timestamps[:n] = tst[:n]

        if ac:
            act = torch.tensor(ac, dtype=torch.int32, device=self.device)
            n = min(self._size, act.shape[0])
            self._access[:n] = act[:n]


# ============================================================
# Aprendizaje y Consolidación Episódica
# ============================================================

class HDCPatternLearner(nn.Module):
    def __init__(self, memory: HDCMemory):
        super().__init__()
        self.memory = memory
        self.factory = memory.factory
    def encode_pattern(self, symbols: List[str], weights: Optional[List[float]] = None) -> torch.Tensor:
        if not symbols:
            return self.factory.random_bipolar()

        vecs = []
        for i, sym in enumerate(symbols):
            hv = self.factory.encode_symbol(sym)
            hv = self.factory.permute(hv, shift=i)
            if weights is not None and i < len(weights):
                hv = hv * float(weights[i])
            vecs.append(hv)

        stacked = torch.stack(vecs, dim=0)
        return _normalize_bipolar(torch.sum(stacked, dim=0))

    def one_shot_store(self, key: str, symbols: List[str], payload: Any, tags: Tuple[str, ...] = (), weight: float = 1.0) -> torch.Tensor:
        hv = self.encode_pattern(symbols)
        self.memory.add(key=key, hypervector=hv, payload=payload, weight=weight, tags=tags)
        return hv

    def bind_fact(self, subject: str, relation: str, obj: str) -> torch.Tensor:
        s = self.factory.encode_symbol(subject)
        r = self.factory.encode_symbol(relation)
        o = self.factory.encode_symbol(obj)
        return self.factory.bundle(self.factory.bind(s, r), self.factory.bind(r, o), self.factory.bind(s, o))


class EpisodicSemanticFusion(nn.Module):
    def __init__(self, memory: HDCMemory, decay_rate: float = 0.005):
        super().__init__()
        self.memory = memory
        self.buffer: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
        self.decay_rate = float(decay_rate)

    def push_to_buffer(self, hypervector: torch.Tensor, metadata: Dict[str, Any]) -> None:
        metadata["timestamp"] = metadata.get("timestamp", time.time())
        hv_device = hypervector.to(device=self.memory.device, dtype=DTYPE)
        self.buffer.append((_normalize_bipolar(hv_device), metadata))

    def consolidate_from_turns(self, turns: List[Dict[str, Any]]) -> int:
        count = 0
        for i, t in enumerate(turns):
            role = t.get("role", "unknown")
            text = t.get("text", "")
            hv = self.memory.factory.encode_symbol(f"{role}:{text[:64]}")
            self.memory.add(
                key=f"turn:{i}",
                hypervector=hv,
                payload={"role": role, "text": text},
                weight=0.6,
                tags=(role,),
            )
            count += 1
        return count

    def consolidate_facts(self, facts: Dict[str, Any]) -> int:
        count = 0
        for k, v in facts.items():
            hv = self.memory.factory.encode_symbol(f"{k}::{v}")
            self.memory.add(
                key=f"fact:{k}",
                hypervector=hv,
                payload={"key": k, "value": v},
                weight=1.0,
                tags=("fact",),
            )
            count += 1
        return count

    def consolidate_context(self) -> torch.Tensor:
        if not self.buffer:
            return torch.zeros(self.memory.dim, dtype=DTYPE, device=self.memory.device)

        accumulated = torch.zeros(self.memory.dim, dtype=DTYPE, device=self.memory.device)
        weights_sum = 0.0
        current_time = time.time()

        for episodic_vector, metadata in self.buffer:
            age = current_time - metadata.get("timestamp", current_time)
            decay_factor = math.exp(-self.decay_rate * age)

            accumulated += episodic_vector * decay_factor
            weights_sum += decay_factor

        if weights_sum > 0:
            accumulated /= weights_sum

        self.buffer = [item for item in self.buffer if (current_time - item[1].get("timestamp", current_time)) < 300]
        return _normalize_bipolar(accumulated)


class HDCStore(nn.Module):
    def save(self, memory: HDCMemory, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(memory.export(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load(self, path: str | Path, dim: int = 10000, seed: Optional[int] = 7, capacity: int = 8192, device: torch.device = DEVICE) -> HDCMemory:
        path = Path(path)
        mem = HDCMemory(dim=dim, seed=seed, capacity=capacity, device=device)
        with path.open("rb") as f:
            payload = pickle.load(f)
        mem.import_(payload)
        return mem


if __name__ == "__main__":
    mem = HDCMemory(dim=4000, seed=7, capacity=8192)
    learner = HDCPatternLearner(mem)
    fusion = EpisodicSemanticFusion(mem, decay_rate=0.005)

    learner.one_shot_store(
        key="objeto_rojo",
        symbols=["objeto", "rojo", "rápido"],
        payload={"label": "objeto rápido y rojo"},
        tags=("perception", "one-shot"),
    )

    learner.one_shot_store(
        key="muro",
        symbols=["muro", "duro", "bloque"],
        payload={"label": "muro resistente"},
        tags=("world",),
    )

    query = learner.encode_pattern(["objeto", "rojo"])
    res = mem.associative_recall(query, threshold=0.10, topk=3)

    print(f"HDC ejecutado en: {mem.device.type.upper()}")
    print("Resultados:")
    if res:
        print(res.key, f"Score: {res.score:.4f}", res.payload)
    else:
        print("sin resultado")

    turns = [
        {"role": "user", "text": "El objeto rojo choca con el muro"},
        {"role": "assistant", "text": "Posible colisión y transferencia de fuerza"},
    ]
    fusion.consolidate_from_turns(turns)
    print("Tamaño memoria:", len(mem))