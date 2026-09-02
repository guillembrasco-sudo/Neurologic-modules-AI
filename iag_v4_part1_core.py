"""
IAG Modular V4 - Parte 1: Núcleo base
=====================================

Esta primera parte establece la infraestructura común para las siguientes capas:

- Configuración global
- Estado persistente
- Historial conversacional
- Memoria de trabajo
- Memoria semántica ligera
- Neuromoduladores digitales
- API base extensible para los módulos cognitivos
- CLI de chat con guardado y carga

Diseñado para ser el punto de anclaje del sistema completo.
Las siguientes partes podrán extender estas clases sin romper la compatibilidad.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import time
from unicodedata import name

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from common import Neuromodulators, set_seed, DEVICE, DTYPE


# ============================================================
# Utilidades Numéricas (PyTorch)
# ============================================================

def _now() -> float:
    return time.time()


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    x_clipped = torch.clamp(x, -40.0, 40.0)
    return torch.sigmoid(x_clipped)


def _stable_norm(x: torch.Tensor, eps: float = 1e-8) -> float:
    return float(torch.sqrt(torch.sum(x * x) + eps))


def _softclip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def _hash_token(token: str, vocab_size: int) -> int:
    # Hash estable y rápido, compatible entre sesiones
    h = 2166136261
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % vocab_size)


def text_to_vector(
    text: str,
    dim: int = 256,
    vocab_size: int = 4096,
    normalize: bool = True,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    if not text:
        return torch.zeros(dim, dtype=DTYPE, device=device)

    tokens = text.lower().split()
    if not tokens:
        return torch.zeros(dim, dtype=DTYPE, device=device)

    # Vectorización completa usando arreglos NumPy/PyTorch acelerados por SIMD/AVX2
    hash_list = [_hash_token(tok, vocab_size) for tok in tokens]
    hashes_t = torch.tensor(hash_list, dtype=torch.int64, device=device)
    
    indices = hashes_t % dim
    signs = torch.where((hashes_t >> 8) & 1 == 1, -1.0, 1.0).to(dtype=DTYPE)
    weights = (1.0 + ((hashes_t >> 16) & 0x7) * 0.125).to(dtype=DTYPE)
    values = signs * weights

    vec = torch.zeros(dim, dtype=DTYPE, device=device)
    vec.scatter_add_(0, indices, values)

    # Mezcla secundaria vectorizada
    seq_idx = torch.arange(len(tokens), device=device, dtype=torch.int64)
    indices2 = (indices * 131 + seq_idx * 17) % dim
    vec.scatter_add_(0, indices2, 0.25 * signs)

    # Rasgos estructurales directos
    features = torch.tensor([
        len(tokens) * 0.05,
        sum(ch.isdigit() for ch in text) * 0.02,
        text.count("?") * 0.1,
        text.count("!") * 0.1
    ], dtype=DTYPE, device=device)
    vec[:4] += features

    if normalize:
        vec = F.normalize(vec, p=2.0, dim=0, eps=1e-8)
    return vec


def vector_to_text_hint(vec: torch.Tensor, topk: int = 8) -> str:
    """
    Devuelve una huella textual de diagnóstico para inspección humana.
    """
    if vec.numel() == 0:
        return "vacío"
    k = min(topk, vec.numel())
    _, idx = torch.topk(torch.abs(vec), k=k)
    parts = [f"{int(i)}:{float(vec[i]):+.3f}" for i in idx.cpu().tolist()]
    return " | ".join(parts)


# ============================================================
# Configuración
# ============================================================

@dataclass(slots=True)
class IAGConfig:
    seed: int = 7
    text_dim: int = 256
    memory_dim: int = 2048
    working_memory_size: int = 64
    semantic_capacity: int = 4096
    history_limit: int = 512
    state_version: str = "v4_part1_pt"
    persona: str = "neutral"
    save_format: str = "pickle"  # pickle | json
    enable_compression: bool = True
    enable_logging: bool = True
    auto_snapshot_every: int = 25


# ============================================================
# Memoria
# ============================================================

@dataclass(slots=True)
class MemoryItem:
    key: str
    value: Any
    vector: torch.Tensor
    timestamp: float
    weight: float = 1.0
    tags: Tuple[str, ...] = ()

    def compact(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "vector": self.vector.detach().cpu().tolist() if isinstance(self.vector, torch.Tensor) else self.vector,
            "timestamp": self.timestamp,
            "weight": self.weight,
            "tags": self.tags,
        }


class WorkingMemory(nn.Module):
    def __init__(self, capacity: int, dim: int):
        super().__init__()
        self.capacity = int(capacity)
        self.dim = int(dim)
        self._items: deque[MemoryItem] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, key: str, value: Any, vector: torch.Tensor, tags: Tuple[str, ...] = ()) -> None:
        vec = vector.to(device=DEVICE, dtype=DTYPE) if isinstance(vector, torch.Tensor) else torch.tensor(vector, device=DEVICE, dtype=DTYPE)
        item = MemoryItem(key=key, value=value, vector=vec, timestamp=_now(), tags=tags)
        self._items.append(item)

    def recent(self, n: int = 8) -> List[MemoryItem]:
        n = min(int(n), len(self._items))
        return [self._items[i] for i in range(len(self._items) - n, len(self._items))]

    def clear(self) -> None:
        self._items.clear()

    def export(self) -> List[Dict[str, Any]]:
        return [item.compact() for item in self._items]

    def import_(self, payload: List[Dict[str, Any]]) -> None:
        self._items.clear()
        for row in payload:
            vec = torch.tensor(row["vector"], dtype=DTYPE, device=DEVICE)
            self._items.append(
                MemoryItem(
                    key=row["key"],
                    value=row["value"],
                    vector=vec,
                    timestamp=float(row["timestamp"]),
                    weight=float(row.get("weight", 1.0)),
                    tags=tuple(row.get("tags", ())),
                )
            )


class SemanticMemory(nn.Module):
    """
    Memoria semántica basada en similitud coseno optimizada para PyTorch en GPU/CPU.
    """
    def __init__(self, capacity: int, dim: int, device: torch.device = DEVICE):
        super().__init__()
        self.capacity = int(capacity)
        self.dim = int(dim)
        self._target_device = device
        self.keys: List[str] = []
        self.values: List[Any] = []
        self.register_buffer("vectors", torch.zeros((self.capacity, self.dim), dtype=DTYPE, device=device))
        self.register_buffer("scores", torch.zeros((self.capacity,), dtype=DTYPE, device=device))
        self.register_buffer("timestamps", torch.zeros((self.capacity,), dtype=torch.float64, device=device))
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

    def add(self, key: str, value: Any, vector: torch.Tensor, score: float = 1.0) -> None:
        idx = self._cursor
        vec = vector.to(device=self.device, dtype=DTYPE)
        v_norm = torch.linalg.vector_norm(vec)
        if v_norm > 1e-8:
            vec = vec / v_norm

        if self._size < self.capacity:
            self.keys.append(key)
            self.values.append(value)
            self._size += 1
        else:
            self.keys[idx] = key
            self.values[idx] = value

        self.vectors[idx] = vec
        self.scores[idx] = float(score)
        self.timestamps[idx] = _now()
        self._cursor = (self._cursor + 1) % self.capacity

    def query(self, vector: torch.Tensor, topk: int = 5) -> List[Tuple[str, Any, float]]:
        if self._size == 0:
            return []

        q = vector.to(device=self.device, dtype=DTYPE)
        qn = torch.linalg.vector_norm(q)
        if qn > 1e-8:
            q = q / qn

        # Producto matriz-vector acelerado
        sims = torch.mv(self.vectors[: self._size], q)
        sims = sims + 0.01 * self.scores[: self._size]

        k = min(topk, self._size)
        vals, idx = torch.topk(sims, k=k)

        out = []
        for v, i in zip(vals.cpu().tolist(), idx.cpu().tolist()):
            out.append((self.keys[i], self.values[i], float(v)))
        return out

    def export(self) -> Dict[str, Any]:
        return {
            "keys": list(self.keys),
            "values": list(self.values),
            "vectors": self.vectors[: self._size].detach().cpu().tolist(),
            "scores": self.scores[: self._size].detach().cpu().tolist(),
            "timestamps": self.timestamps[: self._size].detach().cpu().tolist(),
            "size": self._size,
            "cursor": self._cursor,
        }

    def import_(self, payload: Dict[str, Any]) -> None:
        raw_keys = list(payload.get("keys", []))
        raw_values = list(payload.get("values", []))
        self.keys = raw_keys[:self.capacity]
        self.values = raw_values[:self.capacity]

        raw_vecs = payload.get("vectors", [])
        raw_scores = payload.get("scores", [])
        raw_ts = payload.get("timestamps", [])

        self._size = min(len(self.keys), self.capacity)
        self._cursor = int(payload.get("cursor", self._size % self.capacity)) if self.capacity else 0
        
        self.vectors.zero_()
        self.scores.zero_()
        self.timestamps.zero_()

        if raw_vecs:
            vecs_t = torch.tensor(raw_vecs, dtype=DTYPE, device=self.device)
            n = min(self._size, vecs_t.shape[0])
            self.vectors[:n].copy_(vecs_t[:n])

        if raw_scores:
            scores_t = torch.tensor(raw_scores, dtype=DTYPE, device=self.device)
            n = min(self._size, scores_t.shape[0])
            self.scores[:n] = scores_t[:n]

        if raw_ts:
            ts_t = torch.tensor(raw_ts, dtype=torch.float64, device=self.device)
            n = min(self._size, ts_t.shape[0])
            self.timestamps[:n] = ts_t[:n]


# ============================================================
# Estado general y Persistencia
# ============================================================

@dataclass(slots=True)
class ConversationTurn:
    role: str
    text: str
    timestamp: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def compact(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "timestamp": self.timestamp,
            "meta": self.meta,
        }


@dataclass(slots=True)
class IAGState:
    version: str
    config: IAGConfig
    neuromodulators: Neuromodulators
    turns: List[ConversationTurn] = field(default_factory=list)
    memory_working: List[Dict[str, Any]] = field(default_factory=list)
    memory_semantic: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    goals: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)


class StateStore(nn.Module):
    def __init__(self, root: str | Path = "."):
        super().__init__()
        self.root = Path(root)

    def save(self, state: IAGState, filename: str) -> Path:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if state.config.save_format == "json":
            payload = self._state_to_jsonable(state)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            with path.open("wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load(self, filename: str) -> IAGState:
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return self._jsonable_to_state(payload)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, IAGState):
            return obj
        if isinstance(obj, dict):
            return self._jsonable_to_state(obj)
        raise TypeError(f"Formato de carga no soportado: {type(obj)!r}")

    def _state_to_jsonable(self, state: IAGState) -> Dict[str, Any]:
        return {
            "version": state.version,
            "config": asdict(state.config),
            "neuromodulators": asdict(state.neuromodulators),
            "turns": [t.compact() for t in state.turns],
            "memory_working": state.memory_working,
            "memory_semantic": state.memory_semantic,
            "facts": state.facts,
            "goals": state.goals,
            "statistics": state.statistics,
        }

    def _jsonable_to_state(self, payload: Dict[str, Any]) -> IAGState:
        cfg = IAGConfig(**payload.get("config", {}))
        neu = Neuromodulators(**payload.get("neuromodulators", {}))
        turns = [ConversationTurn(**t) for t in payload.get("turns", [])]

        return IAGState(
            version=str(payload.get("version", "unknown")),
            config=cfg,
            neuromodulators=neu,
            turns=turns,
            memory_working=list(payload.get("memory_working", [])),
            memory_semantic=dict(payload.get("memory_semantic", {})),
            facts=dict(payload.get("facts", {})),
            goals=dict(payload.get("goals", {})),
            statistics=dict(payload.get("statistics", {})),
        )


# ============================================================
# Núcleo Cognitivo Base
# ============================================================

class IAGCoreBase(nn.Module):
    def __init__(self, config: Optional[IAGConfig] = None, device: torch.device = DEVICE):
        super().__init__()
        self.config = config or IAGConfig()
        set_seed(self.config.seed)
        self.device = device

        self.neuromodulators = Neuromodulators()
        self.working_memory = WorkingMemory(self.config.working_memory_size, self.config.memory_dim)
        self.semantic_memory = SemanticMemory(self.config.semantic_capacity, self.config.memory_dim)
        self.history: List[ConversationTurn] = []
        self.facts: Dict[str, Any] = {}
        self.goals: Dict[str, Any] = {}
        self.statistics: Dict[str, Any] = {
            "turns": 0,
            "saves": 0,
            "loads": 0,
            "queries": 0,
            "inferences": 0,
        }
        self.store = StateStore(".")
        self._last_snapshot_turn = 0

    def export_state(self) -> IAGState:
        return IAGState(
            version=self.config.state_version,
            config=self.config,
            neuromodulators=self.neuromodulators,
            turns=list(self.history),
            memory_working=self.working_memory.export(),
            memory_semantic=self.semantic_memory.export(),
            facts=dict(self.facts),
            goals=dict(self.goals),
            statistics=dict(self.statistics),
        )

    def import_state(self, state: IAGState) -> None:
        self.config = state.config
        self.neuromodulators = state.neuromodulators
        self.history = list(state.turns)
        self.working_memory.import_(state.memory_working)
        self.semantic_memory.import_(state.memory_semantic)
        self.facts = dict(state.facts)
        self.goals = dict(state.goals)
        self.statistics = dict(state.statistics)

    def save(self, filename: str = "iag_state.pkl") -> Path:
        self.statistics["saves"] = int(self.statistics.get("saves", 0)) + 1
        return self.store.save(self.export_state(), filename)

    def load(self, filename: str = "iag_state.pkl") -> Path:
        state = self.store.load(filename)
        self.import_state(state)
        self.statistics["loads"] = int(self.statistics.get("loads", 0)) + 1
        return Path(filename)

    def remember(self, key: str, value: Any, tags: Tuple[str, ...] = ()) -> None:
        vec = text_to_vector(f"{key} :: {value}", dim=self.config.memory_dim)
        self.working_memory.add(key, value, vec, tags=tags)
        self.semantic_memory.add(key, value, vec, score=1.0)
        self.facts[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def search_memory(self, query: str, topk: int = 5) -> List[Tuple[str, Any, float]]:
        self.statistics["queries"] = int(self.statistics.get("queries", 0)) + 1
        vec = text_to_vector(query, dim=self.config.memory_dim)
        return self.semantic_memory.query(vec, topk=topk)

    def _append_turn(self, role: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.history.append(ConversationTurn(role=role, text=text, timestamp=_now(), meta=meta or {}))
        if len(self.history) > self.config.history_limit:
            self.history = self.history[-self.config.history_limit :]

    def _extract_intent(self, text: str) -> Dict[str, Any]:
        lower = text.strip().lower()
        intent: Dict[str, Any] = {"raw": text, "type": "chat"}

        is_command = lower.startswith("/")
        clean_text = lower[1:] if is_command else lower

        if clean_text.startswith("recuerda ") or clean_text.startswith("remember "):
            intent["type"] = "remember"
        elif clean_text.startswith("busca ") or clean_text.startswith("search "):
            intent["type"] = "search"
        elif clean_text.startswith("estado") or clean_text.startswith("status"):
            intent["type"] = "status"
        elif "ayuda" in clean_text or clean_text == "help":
            intent["type"] = "help"
        elif clean_text.startswith("objetivo ") or clean_text.startswith("goal "):
            intent["type"] = "goal"
        elif is_command:
            intent["type"] = "command"

        return intent

    @torch.no_grad()
    def process_text(self, text: str) -> str:
        self.statistics["turns"] = int(self.statistics.get("turns", 0)) + 1
        intent = self._extract_intent(text)
        self._append_turn("user", text, meta=intent)

        response = self._route_intent(text, intent)

        self._append_turn("assistant", response, meta={"intent": intent["type"]})

        if self.config.auto_snapshot_every > 0:
            if self.statistics["turns"] - self._last_snapshot_turn >= self.config.auto_snapshot_every:
                self._last_snapshot_turn = self.statistics["turns"]
                self._auto_consolidate()

        return response

    def _route_intent(self, text: str, intent: Dict[str, Any]) -> str:
        t = intent["type"]
        clean_text = text[1:] if text.startswith("/") else text

        if t == "remember":
            content = clean_text.split(maxsplit=1)[1] if len(clean_text.split(maxsplit=1)) > 1 else ""
            if ":" in content:
                k, v = content.split(":", 1)
                self.remember(k.strip(), v.strip())
                return f"Memoria actualizada: {k.strip()}"
            return "Formato: recuerda clave: valor"

        if t == "search":
            query = clean_text.split(maxsplit=1)[1] if len(clean_text.split(maxsplit=1)) > 1 else ""
            results = self.search_memory(query, topk=4)
            if not results:
                return "Sin coincidencias en memoria."
            lines = [f"{k} ({s:+.3f}) = {v}" for k, v, s in results]
            return "Coincidencias:\n" + "\n".join(lines)

        if t == "status":
            return self.status_report()

        if t == "help":
            return self.help_text()

        if t == "goal":
            goal = clean_text.split(maxsplit=1)[1] if len(clean_text.split(maxsplit=1)) > 1 else ""
            gid = f"g{len(self.goals) + 1}"
            self.goals[gid] = {"goal": goal, "created": _now(), "state": "active"}
            self.neuromodulators.boost("dopamina", 0.03)
            return f"Objetivo creado: {gid}"

        if t == "command":
            return f"Comando no reconocido por el sistema base: '{text}'"

        return self.generate_reply(text, intent)

    def generate_reply(self, text: str, intent: Dict[str, Any]) -> str:
        query_vec = text_to_vector(text, dim=self.config.memory_dim)
        neighbors = self.semantic_memory.query(query_vec, topk=3)
        self.statistics["inferences"] = int(self.statistics.get("inferences", 0)) + 1

        if neighbors:
            best_key, best_value, best_score = neighbors[0]
            if best_score > 0.35:
                self.neuromodulators.boost("confianza", 0.01)
                return f"Conecto con memoria: {best_key} -> {best_value}"

        h = vector_to_text_hint(query_vec, topk=5)
        return (
            "Procesado por el núcleo base PyTorch. "
            f"Huella={h}. "
            "Listo para conectar los siguientes módulos."
        )

    def status_report(self) -> str:
        nm = self.neuromodulators
        return (
            f"Estado IAG Base (PyTorch en {DEVICE.type.upper()})\n"
            f"- turns: {self.statistics.get('turns', 0)}\n"
            f"- memory_working: {len(self.working_memory)}\n"
            f"- memory_semantic: {len(self.semantic_memory)}\n"
            f"- goals: {len(self.goals)}\n"
            f"- dopamina: {nm.dopamina:.2f}\n"
            f"- serotonina: {nm.serotonina:.2f}\n"
            f"- noradrenalina: {nm.noradrenalina:.2f}\n"
            f"- acetilcolina: {nm.acetilcolina:.2f}\n"
            f"- adrenalina: {nm.adrenalina:.2f}\n"
            f"- curiosidad: {nm.curiosidad:.2f}\n"
            f"- fatiga: {nm.fatiga:.2f}\n"
            f"- confianza: {nm.confianza:.2f}"
        )

    def help_text(self) -> str:
        return (
            "Comandos:\n"
            "/help\n"
            "/status\n"
            "/save archivo.pkl\n"
            "/load archivo.pkl\n"
            "/remember clave: valor\n"
            "/recall clave\n"
            "/search texto\n"
            "/goal objetivo\n"
            "/history\n"
            "/mem\n"
            "/exit"
        )

    def show_history(self, n: int = 20) -> str:
        slice_ = self.history[-int(n):]
        if not slice_:
            return "Sin historial."
        return "\n".join([f"[{t.role}] {t.text}" for t in slice_])

    def show_memory(self, n: int = 8) -> str:
        items = self.working_memory.recent(n=n)
        if not items:
            return "Memoria de trabajo vacía."
        return "\n".join([f"{item.key} -> {item.value}" for item in items])

    def _auto_consolidate(self) -> None:
        recent = self.working_memory.recent(n=min(8, len(self.working_memory)))
        if not recent:
            return
        for item in recent:
            self.semantic_memory.add(item.key, item.value, item.vector, score=item.weight)
        self.neuromodulators.boost("serotonina", 0.005)
        self.neuromodulators.boost("fatiga", 0.01)
        self.neuromodulators.clamp()


# ============================================================
# Chat interactivo
# ============================================================

class IAGChatCLI(nn.Module):
    def __init__(self, brain: IAGCoreBase):
        super().__init__()
        self.brain = brain

    def run(self) -> None:
        print(f"IAG Modular V4 - Parte 1 (PyTorch en {DEVICE.type.upper()})")
        print("Escribe /help para ver comandos.\n")

        while True:
            try:
                user = input("Tu> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo.")
                break

            if not user:
                continue

            if user in {"/exit", "exit", "quit"}:
                print("Saliendo.")
                break

            if user.startswith("/save"):
                parts = user.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "iag_state.pkl"
                path = self.brain.save(filename)
                print(f"Guardado en: {path}")
                continue

            if user.startswith("/load"):
                parts = user.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else "iag_state.pkl"
                path = self.brain.load(filename)
                print(f"Cargado desde: {path}")
                continue

            if user == "/history":
                print(self.brain.show_history())
                continue

            if user == "/mem":
                print(self.brain.show_memory())
                continue

            if user.startswith("/recall"):
                parts = user.split(maxsplit=1)
                key = parts[1] if len(parts) > 1 else ""
                print(self.brain.recall(key, default="No encontrado"))
                continue

            response = self.brain.process_text(user)
            print("IAG>", response)


def build_core(seed: int = 7) -> IAGCoreBase:
    cfg = IAGConfig(seed=seed)
    return IAGCoreBase(cfg)


if __name__ == "__main__":
    core = build_core()
    cli = IAGChatCLI(core)
    cli.run()