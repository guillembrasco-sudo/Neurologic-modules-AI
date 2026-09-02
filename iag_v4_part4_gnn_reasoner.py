"""
IAG Modular V4 - Parte 4
GNN / grafo relacional para inferencia estructurada.

Objetivos:
- Representar entidades, relaciones y eventos como nodos/aristas
- Propagar mensajes entre nodos
- Inferir dependencias y posibles efectos
- Mantener compatibilidad con salidas HDC / percepción
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import numpy as np

logger = logging.getLogger("braincore")


# ============================================================
# Dispositivo y Tipos Globales
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilidades de Tensores
# ============================================================

def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def _as_vec(x: Any, dim: int, device: torch.device = DEVICE) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        v = x.to(device=device, dtype=DTYPE)
    else:
        v = torch.tensor(x, dtype=DTYPE, device=device)

    if v.ndim == 0:
        v = v.unsqueeze(0)

    if v.numel() < dim:
        v = F.pad(v, (0, dim - v.numel()))
    else:
        v = v[:dim]
    return v


# ============================================================
# Estructuras del Grafo
# ============================================================

@dataclass
class NodeState:
    node_id: str
    kind: str
    feature: torch.Tensor
    belief: float = 0.5
    salience: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeState:
    src: str
    dst: str
    relation: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphInference:
    target: str
    score: float
    explanation: str
    evidence: List[str] = field(default_factory=list)


# ============================================================
# Grafo Cognitivo
# ============================================================

class CognitiveGraph(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 128, seed: Optional[int] = 7, device: torch.device = DEVICE):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self._target_device = device

        self.nodes: Dict[str, NodeState] = {}
        self.edges: List[EdgeState] = []

        # Inicialización determinista de pesos mediante PyTorch Generator
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        self.W_msg_down = nn.Parameter((torch.randn(self.hidden, self.dim, generator=gen) * 0.05).to(device=self.device, dtype=DTYPE))
        self.W_msg_up = nn.Parameter((torch.randn(self.dim, self.hidden, generator=gen) * 0.05).to(device=self.device, dtype=DTYPE))
        self.W_upd = nn.Parameter((torch.randn(self.dim, self.dim, generator=gen) * 0.05).to(device=self.device, dtype=DTYPE))
        self.W_gate = nn.Parameter((torch.randn(self.dim, 2 * self.dim, generator=gen) * 0.05).to(device=self.device, dtype=DTYPE))

        self.rel_bias: Dict[str, float] = {}

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return self._target_device

    # -------------------------
    # Construcción
    # -------------------------

    def add_node(
        self,
        node_id: str,
        kind: str,
        feature: Any,
        belief: float = 0.5,
        salience: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        feat = _normalize(_as_vec(feature, self.dim, device=self.device))
        self.nodes[node_id] = NodeState(
            node_id=node_id,
            kind=kind,
            feature=feat,
            belief=float(torch.clamp(torch.tensor(belief), 0.0, 1.0).item()),
            salience=float(max(0.0, salience)),
            metadata=metadata or {},
        )

    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Los nodos src y dst deben existir antes de crear la arista.")
        self.edges.append(
            EdgeState(
                src=src,
                dst=dst,
                relation=relation,
                weight=float(weight),
                metadata=metadata or {},
            )
        )

    def add_fact(self, subject: str, relation: str, obj: str, subj_feat: Any = None, obj_feat: Any = None) -> None:
        if subject not in self.nodes:
            self.add_node(subject, "entity", subj_feat if subj_feat is not None else self._symbol_feature(subject))
        if obj not in self.nodes:
            self.add_node(obj, "entity", obj_feat if obj_feat is not None else self._symbol_feature(obj))
        self.add_edge(subject, obj, relation, weight=1.0)

    def _symbol_feature(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim, dtype=DTYPE, device=self.device)
        b = text.encode("utf-8")
        for i, ch in enumerate(b):
            v[(ch + 17 * i) % self.dim] += 1.0
        return _normalize(v)

    # -------------------------
    # Mensajería / Paso de Mensajes
    # -------------------------

    def _incoming(self, node_id: str) -> List[EdgeState]:
        return [e for e in self.edges if e.dst == node_id]

    def _outgoing(self, node_id: str) -> List[EdgeState]:
        return [e for e in self.edges if e.src == node_id]

    def _build_adjacency_matrix(self, num_nodes: int, id_to_idx: Dict[str, int]) -> torch.Tensor:
        dev = self.device
        valid_edges = [e for e in self.edges if e.src in id_to_idx and e.dst in id_to_idx]
        
        if not valid_edges:
            indices = torch.empty((2, 0), dtype=torch.long, device=dev)
            values = torch.empty((0,), dtype=DTYPE, device=dev)
            return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=dev)

        edge_dict: Dict[Tuple[int, int], float] = {}
        for e in valid_edges:
            src_i = id_to_idx[e.src]
            dst_i = id_to_idx[e.dst]
            w = e.weight * self.rel_bias.get(e.relation, 1.0)
            key = (dst_i, src_i)
            # Selecciona el peso máximo entre aristas paralelas en lugar de sumar
            edge_dict[key] = max(edge_dict.get(key, 0.0), w)

        if not edge_dict:
            indices = torch.empty((2, 0), dtype=torch.long, device=dev)
            values = torch.empty((0,), dtype=DTYPE, device=dev)
            return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=dev)

        indices_list = list(edge_dict.keys())
        values_list = list(edge_dict.values())

        indices = torch.tensor(indices_list, dtype=torch.long, device=dev).t()
        values = torch.tensor(values_list, dtype=DTYPE, device=dev)

        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=dev).coalesce()

    def propagate(self, steps: int = 1) -> None:
        """
        Propagación matricial dispersa global (Gated Graph Conv) sin bucles Python en aristas.
        """
        if not self.nodes:
            return

        target_dev = self.device
        node_ids = list(self.nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        for nid in node_ids:
            if self.nodes[nid].feature.device != target_dev:
                self.nodes[nid].feature = self.nodes[nid].feature.to(device=target_dev, dtype=DTYPE)

        # Forzar la alineación de dispositivo de las características alojadas en los nodos
        X = torch.stack([self.nodes[nid].feature for nid in node_ids], dim=0)
        B = torch.tensor([self.nodes[nid].belief for nid in node_ids], dtype=DTYPE, device=target_dev).unsqueeze(1)
    
        if not self.edges:
            # Transformación interna sin agregación de aristas
            for _ in range(steps):
                X_proj = torch.matmul(torch.matmul(X, self.W_msg_down.T), self.W_msg_up.T)
                X = _normalize(X_proj)
            for i, nid in enumerate(node_ids):
                self.nodes[nid].feature = X[i]
            return

        # Construir Matriz de Adyacencia Ponderada Dispersa A (N, N)
        A = self._build_adjacency_matrix(num_nodes, id_to_idx)

        # Grado de entrada para normalización
        deg = torch.sparse.sum(A, dim=1).to_dense().unsqueeze(1).clamp(min=1.0)

        for _ in range(steps):
            X_proj = torch.matmul(torch.matmul(X, self.W_msg_down.T), self.W_msg_up.T)
            X_weighted = X_proj * B
            Agg_Messages = torch.sparse.mm(A, X_weighted) / deg

            Combined = torch.cat([Agg_Messages, X], dim=1)
            Z = torch.sigmoid(torch.matmul(Combined, self.W_gate.T))
            Candidate = torch.tanh(torch.matmul(Agg_Messages, self.W_upd.T))

            X = _normalize((1.0 - Z) * X + Z * Candidate)
            Incoming_Beliefs = torch.sparse.mm(A, B) / deg
            B = (0.7 * B + 0.3 * Incoming_Beliefs).clamp(0.0, 1.0)

        for i, nid in enumerate(node_ids):
            self.nodes[nid].feature = X[i].detach()
            self.nodes[nid].belief = float(B[i].item())

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse)
        # Sobrescribimos el hook interno de PyTorch para garantizar migración de los nodos en Python dict
        for node in self.nodes.values():
            if isinstance(node.feature, torch.Tensor):
                node.feature = fn(node.feature)
        return self
    # -------------------------
    # Inferencia Relacional
    # -------------------------

    def infer_related(self, query_node: str, topk: int = 5) -> List[GraphInference]:
        if query_node not in self.nodes:
            return []

        q = self.nodes[query_node]
        candidates: List[GraphInference] = []

        for edge in self._outgoing(query_node):
            dst = self.nodes[edge.dst]
            sim = (q.feature @ dst.feature).item()
            score = float(sim * edge.weight * (0.5 + dst.belief))
            candidates.append(
                GraphInference(
                    target=edge.dst,
                    score=score,
                    explanation=f"{query_node} -[{edge.relation}]-> {edge.dst}",
                    evidence=[query_node, edge.dst, edge.relation],
                )
            )

        for edge in self._incoming(query_node):
            src = self.nodes[edge.src]
            sim = (src.feature @ q.feature).item()
            score = float(sim * edge.weight * (0.5 + src.belief))
            candidates.append(
                GraphInference(
                    target=edge.src,
                    score=score,
                    explanation=f"{edge.src} -[{edge.relation}]-> {query_node}",
                    evidence=[edge.src, query_node, edge.relation],
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[: int(topk)]

    def infer_causal_chain(self, start: str, max_depth: int = 3) -> List[GraphInference]:
        if start not in self.nodes:
            return []

        frontier = [(start, 0, [start], 1.0)]
        seen = {start}
        results: List[GraphInference] = []

        while frontier:
            node_id, depth, path, score = frontier.pop(0)
            if depth >= max_depth:
                continue

            for edge in self._outgoing(node_id):
                if edge.dst in path:
                    continue

                dst = self.nodes[edge.dst]
                new_score = score * edge.weight * (0.5 + dst.belief)
                new_path = path + [edge.dst]

                results.append(
                    GraphInference(
                        target=edge.dst,
                        score=float(new_score),
                        explanation=" -> ".join(new_path),
                        evidence=new_path + [edge.relation],
                    )
                )
                frontier.append((edge.dst, depth + 1, new_path, new_score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def predict_effect(self, subject: str, relation: str, topk: int = 5) -> List[GraphInference]:
        if subject not in self.nodes:
            return []

        candidates = []
        for edge in self._outgoing(subject):
            if edge.relation != relation:
                continue
            dst = self.nodes[edge.dst]
            score = float(edge.weight * (0.5 + dst.belief) * (0.5 + self.nodes[subject].belief))
            candidates.append(
                GraphInference(
                    target=edge.dst,
                    score=score,
                    explanation=f"Si {subject} [{relation}] {edge.dst}, entonces {edge.dst} queda activado",
                    evidence=[subject, relation, edge.dst],
                )
            )
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[: int(topk)]

    # -------------------------
    # Integración
    # -------------------------

    def ingest_triplets(self, triplets: List[Tuple[str, str, str]]) -> int:
        count = 0
        for s, r, o in triplets:
            self.add_fact(s, r, o)
            count += 1
        return count

    def attach_features_from_map(self, feature_map: Dict[str, Any]) -> int:
        count = 0
        for node_id, feat in feature_map.items():
            if node_id in self.nodes:
                self.nodes[node_id].feature = _normalize(_as_vec(feat, self.dim, device=self.device))
                count += 1
        return count

    # -------------------------
    # Diagnóstico
    # -------------------------

    def summary(self) -> str:
        kinds: Dict[str, int] = {}
        for n in self.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        rels: Dict[str, int] = {}
        for e in self.edges:
            rels[e.relation] = rels.get(e.relation, 0) + 1

        return (
            f"CognitiveGraph (Device: {self.device.type.upper()})\n"
            f"- nodes: {len(self.nodes)}\n"
            f"- edges: {len(self.edges)}\n"
            f"- kinds: {kinds}\n"
            f"- relations: {rels}"
        )

    # -------------------------
    # Persistencia
    # -------------------------

    def export(self) -> Dict[str, Any]:
        serialized_nodes = {
            nid: {
                "node_id": n.node_id,
                "kind": n.kind,
                "feature": n.feature.detach().cpu().tolist(),
                "belief": float(n.belief),
                "salience": float(n.salience),
                "metadata": n.metadata,
            }
            for nid, n in self.nodes.items()
        }

        serialized_edges = [
            {
                "src": e.src,
                "dst": e.dst,
                "relation": e.relation,
                "weight": float(e.weight),
                "metadata": e.metadata,
            }
            for e in self.edges
        ]

        return {
            "dim": self.dim,
            "hidden": self.hidden,
            "nodes": serialized_nodes,
            "edges": serialized_edges,
            "rel_bias": dict(self.rel_bias),
            "weights": {
                "W_msg_down": self.W_msg_down.detach().cpu(),
                "W_msg_up": self.W_msg_up.detach().cpu(),
                "W_upd": self.W_upd.detach().cpu(),
                "W_gate": self.W_gate.detach().cpu(),
            }
        }

    def import_(self, payload: Dict[str, Any]) -> None:
        self.dim = int(payload.get("dim", self.dim))
        self.hidden = int(payload.get("hidden", self.hidden))

        self.nodes = {}
        for nid, n in payload.get("nodes", {}).items():
            feat_tensor = _normalize(torch.tensor(n["feature"], dtype=DTYPE, device=self.device))
            self.nodes[nid] = NodeState(
                node_id=n["node_id"],
                kind=n["kind"],
                feature=feat_tensor,
                belief=float(n.get("belief", 0.5)),
                salience=float(n.get("salience", 1.0)),
                metadata=dict(n.get("metadata", {})),
            )

        self.edges = []
        for e in payload.get("edges", []):
            self.edges.append(
                EdgeState(
                    src=e["src"],
                    dst=e["dst"],
                    relation=e["relation"],
                    weight=float(e.get("weight", 1.0)),
                    metadata=dict(e.get("metadata", {})),
                )
            )

        if "weights" in payload:
            w = payload["weights"]
            with torch.no_grad():
                self.W_msg_down.copy_(w["W_msg_down"].to(self.device))
                self.W_msg_up.copy_(w["W_msg_up"].to(self.device))
                self.W_upd.copy_(w["W_upd"].to(self.device))
                self.W_gate.copy_(w["W_gate"].to(self.device))

        self.rel_bias = {str(k): float(v) for k, v in payload.get("rel_bias", {}).items()}


class GraphStore(nn.Module):
    def save(self, graph: CognitiveGraph, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(graph.export(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load(self, path: str | Path, device: torch.device = DEVICE) -> CognitiveGraph:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        g = CognitiveGraph(dim=int(payload.get("dim", 256)), hidden=int(payload.get("hidden", 128)), device=device)
        g.import_(payload)
        return g


if __name__ == "__main__":
    g = CognitiveGraph(dim=256, hidden=128, seed=7)

    g.add_node("objeto_rojo", "entity", torch.randn(256))
    g.add_node("muro", "entity", torch.randn(256))
    g.add_node("choque", "event", torch.randn(256), belief=0.8)

    g.add_edge("objeto_rojo", "choque", "causa", weight=0.9)
    g.add_edge("choque", "muro", "impacta", weight=0.95)

    g.propagate(steps=2)

    print(g.summary())
    print("\nInferencia de relaciones con 'choque':")
    for inf in g.infer_related("choque", topk=3):
        print(f"  - Target: {inf.target} | Score: {inf.score:.4f} | {inf.explanation}")