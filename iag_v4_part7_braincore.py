"""
IAG Modular V4 - Parte 7
BrainCore: integrador cognitivo central.

Une las partes anteriores:
- Parte 1: núcleo base
- Parte 2: percepción multimodal
- Parte 3: HDC
- Parte 4: GNN
- Parte 5: LNN
- Parte 6: Atractores

Responsabilidades:
- Recibir entrada multimodal
- Convertirla en representación compartida
- Escribir en memoria
- Construir/actualizar grafo
- Ejecutar LNN para control adaptativo
- Generar hipótesis creativas
- Producir respuesta unificada
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.util
import sys
import time
import pickle
import math
import hmac, hashlib

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from common import DEVICE, DTYPE


BASE_DIR = Path("E:/IA/Brain")
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def _import_from_file(module_name: str, file_name: str):
    path = BASE_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Falta el archivo requerido: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar: {file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


p1 = _import_from_file("iag_v4_part1_core", "iag_v4_part1_core.py")
p2 = _import_from_file("iag_v4_part2_perception", "iag_v4_part2_perception.py")
p3 = _import_from_file("iag_v4_part3_hdc_memory", "iag_v4_part3_hdc_memory.py")
p4 = _import_from_file("iag_v4_part4_gnn_reasoner", "iag_v4_part4_gnn_reasoner.py")
p5 = _import_from_file("iag_v4_part5_lnn", "iag_v4_part5_lnn.py")
p6 = _import_from_file("iag_v4_part6_attractors_creativity", "iag_v4_part6_attractors_creativity.py")


# ============================================================
# Utilidades de Tensores
# ============================================================

def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=DTYPE, device=DEVICE)
    else:
        x = x.to(device=DEVICE, dtype=DTYPE)
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=DTYPE, device=DEVICE)
    else:
        x = x.to(device=DEVICE, dtype=DTYPE)
    x = x.reshape(-1)
    if x.numel() < dim:
        return F.pad(x, (0, dim - x.numel()))
    if x.numel() > dim:
        return x[:dim]
    return x


def _text_hint(vec: torch.Tensor, topk: int = 6) -> str:
    if vec.numel() == 0:
        return "vacío"
    topk_val = min(topk, vec.numel())
    _, idx = torch.topk(torch.abs(vec), topk_val)
    return " | ".join(f"{int(i.item())}:{float(vec[i].item()):+.3f}" for i in idx)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Calcula la similitud del coseno entre dos vectores sobre PyTorch."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=DTYPE, device=DEVICE)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=DTYPE, device=DEVICE)
    a_norm = F.normalize(a.reshape(-1), p=2.0, dim=0, eps=eps)
    b_norm = F.normalize(b.reshape(-1), p=2.0, dim=0, eps=eps)
    return float(torch.dot(a_norm, b_norm).item())


# ============================================================
# Configuraciones y Contenedor de Decisiones Cognitivas
# ============================================================

@dataclass(slots=True)
class BrainConfig:
    seed: int = 7
    perception_dim: int = 1024
    hdc_dim: int = 49152
    graph_dim: int = 1024
    lnn_input_dim: int = 1024
    lnn_hidden_dim: int = 1024
    lnn_output_dim: int = 512
    attractor_dim: int = 1024
    history_limit: int = 4196
    save_format: str = "pickle"
    auto_consolidate_every: int = 5


@dataclass(slots=True)
class BrainDecision:
    """Representación estructural inmutable de la salida holística del núcleo cognitivo."""
    actions: torch.Tensor = field(default_factory=lambda: torch.zeros(128, dtype=DTYPE, device=torch.device("cpu")))
    cognitive_state: torch.Tensor = field(default_factory=lambda: torch.zeros(256, dtype=DTYPE, device=torch.device("cpu")))
    attention_focus: str = "root"

    answer: str = ""
    mode: str = "reason+memory"
    confidence: float = 0.5
    signals: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ============================================================
# Core Integrador (BrainCore)
# ============================================================

class BrainCore(nn.Module):
    """
    Orquestador Central y Bus de Datos de la Arquitectura IAG Modular V4.
    Gobierna y sincroniza HDC, Memorias, LNN, GNN y Sistemas Atractores sobre PyTorch.
    """
    def __init__(self, cfg: Optional[BrainConfig] = None, device: torch.device = DEVICE):
        super().__init__()
        self.cfg = cfg or BrainConfig()
        self.device = device

        self._lock = threading.RLock()

        base_cfg = p1.IAGConfig(seed=self.cfg.seed, text_dim=256, memory_dim=self.cfg.hdc_dim)
        self.base = p1.IAGCoreBase(base_cfg, device=self.device)

        self.perception = p2.PerceptionHub(device=self.device)

        self.hdc = p3.HDCMemory(dim=self.cfg.hdc_dim, seed=self.cfg.seed, capacity=8192, device=self.device)
        self.hdc_learner = p3.HDCPatternLearner(self.hdc)
        self.hdc_fusion = p3.EpisodicSemanticFusion(self.hdc)

        self.hdc_dim = self.cfg.hdc_dim
        self.dense_dim = self.cfg.graph_dim

        self.graph = p4.CognitiveGraph(dim=self.cfg.graph_dim, hidden=128, seed=self.cfg.seed, device=self.device)

        lnn_cfg = p5.LNNConfig(
            input_dim=self.cfg.lnn_input_dim,
            hidden_dim=self.cfg.lnn_hidden_dim,
            output_dim=self.cfg.lnn_output_dim,
            seed=self.cfg.seed,
            integration="rk4",
        )
        self.lnn = p5.LiquidNeuralNetwork(cfg=lnn_cfg, neuromodulators=self.base.neuromodulators, device=self.device)

        attr_cfg = p6.AttractorConfig(latent_dim=self.cfg.attractor_dim, attractor_count=16, seed=self.cfg.seed)
        self.creativity = p6.AttractorEngine(cfg=attr_cfg, neuromodulators=self.base.neuromodulators, device=self.device)

        gen_proj = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            gen_proj.manual_seed(self.cfg.seed)

        # Registro formal de buffers de proyección
        std_down = 1.0 / math.sqrt(self.hdc_dim)
        W_down = (torch.randn(self.dense_dim, self.hdc_dim, generator=gen_proj) * std_down).to(dtype=DTYPE)
        self.register_buffer("W_downproject", W_down.to(self.device))

        std_up = 1.0 / math.sqrt(self.dense_dim)
        W_up = (torch.randn(self.hdc_dim, self.dense_dim, generator=gen_proj) * std_up).to(dtype=DTYPE)
        self.register_buffer("W_upproject", W_up.to(self.device))

        self.turn_counter = 0
        self.last_latent = torch.zeros((self.cfg.perception_dim,), dtype=DTYPE, device=self.device)
        self.last_hdc = torch.zeros((self.cfg.hdc_dim,), dtype=DTYPE, device=self.device)
        self.last_lnn_out = torch.zeros((self.cfg.lnn_output_dim,), dtype=DTYPE, device=self.device)

        self.history: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {
            "turns": 0,
            "memory_writes": 0,
            "graph_updates": 0,
            "creative_calls": 0,
            "lnn_steps": 0,
            "reason_calls": 0,
        }

    def _compress_hdc_to_dense(self, hdc_vector: torch.Tensor) -> torch.Tensor:
        """Comprime un hipervector HDC a dimensiones densas para LNN/GNN."""
        if not isinstance(hdc_vector, torch.Tensor):
            hdc_vector = torch.tensor(hdc_vector, dtype=DTYPE, device=self.device)
        else:
            hdc_vector = hdc_vector.to(device=self.device, dtype=DTYPE)
        
        dense_mapped = F.linear(hdc_vector, self.W_downproject)
        return torch.tanh(dense_mapped)

    def _expand_dense_to_hdc(self, dense_vector: torch.Tensor) -> torch.Tensor:
        """Expande y discretiza un vector denso para transformarlo en un hipervector compatible."""
        if not isinstance(dense_vector, torch.Tensor):
            dense_vector = torch.tensor(dense_vector, dtype=DTYPE, device=self.device)
        else:
            dense_vector = dense_vector.to(device=self.device, dtype=DTYPE)
            
        raw_hdc = F.linear(dense_vector, self.W_upproject)
        return torch.where(raw_hdc >= 0.0, 1.0, -1.0).to(dtype=DTYPE)

    def observe(
        self,
        text: str,
        image: Any = None,
        sequence: Any = None,
        vector_data: Optional[torch.Tensor] = None,
    ) -> BrainDecision:
        latents = []

        if text is not None and len(text) > 0:
            lat_text = self.perception.perceive_text(text)
            latents.append(_pad_or_trim(lat_text, self.cfg.perception_dim))

        if image is not None:
            lat_img = self.perception.perceive_image(image)
            latents.append(_pad_or_trim(lat_img, self.cfg.perception_dim))

        if sequence is not None:
            lat_seq = self.perception.perceive_sequence(sequence)
            latents.append(_pad_or_trim(lat_seq, self.cfg.perception_dim))

        if not latents:
            latent = torch.zeros((self.cfg.perception_dim,), dtype=DTYPE, device=self.device)
        else:
            stacked = torch.stack(latents, dim=0)
            latent = _normalize(torch.mean(stacked, dim=0))

        self.last_latent = latent.clone()

        if vector_data is not None:
            perceptual_vector = _normalize(_pad_or_trim(vector_data, self.cfg.hdc_dim))
        else:
            perceptual_vector = self._expand_dense_to_hdc(_pad_or_trim(latent, self.dense_dim))

        # 2. Consolidación en la memoria episódica a corto plazo
        self.hdc_fusion.push_to_buffer(perceptual_vector, {"timestamp": time.time()})
        semantic_context = self.hdc_fusion.consolidate_context()

        low_dim_stimulus = self._compress_hdc_to_dense(semantic_context)
        lnn_state = self.lnn.step_continuous_adaptive(low_dim_stimulus)

        obs_id = f"obs_{self.turn_counter}"
        self.graph.add_node(obs_id, "observation_root", _pad_or_trim(latent, self.cfg.graph_dim), belief=1.0, salience=1.0)

        chunk_size = 10
        num_nodes = min(5, lnn_state.numel() // chunk_size)
        for i in range(num_nodes):
            node_id = f"latent_node_{self.turn_counter}_{i}"
            feat = _pad_or_trim(lnn_state[i * chunk_size : (i + 1) * chunk_size], self.cfg.graph_dim)
            self.graph.add_node(node_id, "latent_trajectory", feat, belief=0.5, salience=1.0)

            if obs_id in self.graph.nodes:
                self.graph.add_edge(obs_id, node_id, "dynamic_state", weight=0.7)

        self.graph.propagate(steps=1)

        if self.graph.nodes:
            feats = torch.stack([node.feature * node.belief for node in self.graph.nodes.values()])
            graph_context = torch.mean(feats, dim=0)
            graph_context = _pad_or_trim(graph_context, self.cfg.attractor_dim)
        else:
            graph_context = torch.zeros(self.cfg.attractor_dim, dtype=DTYPE, device=self.device)

        forcing_field = _normalize(_pad_or_trim(lnn_state, self.cfg.attractor_dim) + graph_context)
        stabilized_state = self.creativity.evolve(forcing_field, steps=5)

        action_space = torch.tanh(stabilized_state)

        if self.graph.nodes:
            dominant_node = max(self.graph.nodes.items(), key=lambda item: float(item[1].belief))[0]
        else:
            dominant_node = "root"

        return BrainDecision(
            actions=action_space,
            cognitive_state=stabilized_state,
            attention_focus=dominant_node,
        )

    def _latent_to_graph_features(self, latent: torch.Tensor) -> torch.Tensor:
        return _pad_or_trim(latent, self.cfg.graph_dim)

    def _latent_to_lnn_input(self, latent: torch.Tensor, hdc_vec: torch.Tensor) -> torch.Tensor:
        a = _pad_or_trim(latent, self.cfg.lnn_input_dim)
        b = self._compress_hdc_to_dense(hdc_vec)
        return _normalize(0.65 * a + 0.35 * b)

    def _latent_to_attractor_input(self, latent: torch.Tensor) -> torch.Tensor:
        return _pad_or_trim(latent, self.cfg.attractor_dim)

    def _latent_to_hdc(self, latent: torch.Tensor, text: str = "") -> torch.Tensor:
        key = f"obs::{text[:64]}"
        hv = self.hdc_learner.encode_pattern([key, f"latent::{_text_hint(latent, 4)}"])
        return hv

    def remember_observation(self, text: str, latent: torch.Tensor, hdc_vec: torch.Tensor) -> None:
        latent_slice = latent[:16].detach().cpu().tolist()
        self.base.remember(
            key=f"obs_{self.turn_counter}",
            value={"text": text, "latent": latent_slice},
            tags=("observation",),
        )
        self.hdc.add(
            key=f"obs_{self.turn_counter}",
            hypervector=hdc_vec,
            payload={"text": text, "latent": latent_slice},
            weight=1.0,
            tags=("observation",),
        )
        self.statistics["memory_writes"] += 1

    def update_graph_from_text(self, text: str, latent: torch.Tensor) -> None:
        obs_id = f"obs_{self.turn_counter}"
        feat = self._latent_to_graph_features(latent)
        self.graph.add_node(obs_id, "observation", feat, belief=0.6, salience=1.0, metadata={"text": text})

        words = [w.strip(".,;:!?") for w in text.lower().split() if w.strip(".,;:!?")]
        tokens = []
        for w in words[:8]:
            node_id = f"tok::{w}"
            if node_id not in self.graph.nodes:
                self.graph.add_node(
                    node_id,
                    "token",
                    self.graph._symbol_feature(w),
                    belief=0.4,
                    salience=0.7,
                )
            tokens.append(node_id)
            self.graph.add_edge(obs_id, node_id, "mentions", weight=0.65)

        for a, b in zip(tokens, tokens[1:]):
            try:
                self.graph.add_edge(a, b, "follows", weight=0.25)
            except Exception:
                pass

        self.graph.propagate(steps=1)
        self.statistics["graph_updates"] += 1

    # ============================================================
    # Gating y Decodificación Vectorial Adaptativa
    # ============================================================

    def _should_generate_speech(self, lnn_output: torch.Tensor, input_latent: torch.Tensor, max_sim: float) -> bool:
        """
        Evalúa si la red debe hablar basándose en tres fuerzas:
        1. Energía del input (excitación por entrada de usuario)
        2. Estado interno de la LNN
        3. Afinidad con recuerdos en memoria
        """
        input_energy = float(torch.linalg.vector_norm(input_latent.reshape(-1)).item())
        lnn_drive = float(torch.tanh(torch.mean(lnn_output[:4])).detach().cpu().item())

        speech_drive = (0.4 * input_energy) + (0.3 * lnn_drive) + (0.3 * max_sim)

        nm = self.base.neuromodulators
        threshold = 0.18 - (0.05 * nm.dopamina) - (0.05 * nm.curiosidad)

        return speech_drive > threshold

    def _recurrent_refinement_loop(
        self,
        prompt_vector: torch.Tensor,
        raw_matches: List[Any],
        max_iterations: int = 3,
        initial_threshold: float = 0.25,
    ) -> List[str]:
        """
        Bucle de refinamiento recurrente con relajación adaptativa de umbral.
        """
        if not raw_matches:
            return []

        dense_prompt = self._compress_hdc_to_dense(prompt_vector)
        threshold = initial_threshold

        for _ in range(max_iterations):
            filtered_candidates = []

            for match in raw_matches:
                payload = getattr(match, "payload", {})
                text_content = payload.get("text", str(match.key)) if isinstance(payload, dict) else str(match.key)

                candidate_hdc = getattr(match, "hypervector", None)
                if candidate_hdc is not None:
                    candidate_dense = self._compress_hdc_to_dense(candidate_hdc)
                else:
                    candidate_dense = self._latent_to_graph_features(self.perception.perceive_text(text_content))

                similarity = _cosine_similarity(dense_prompt, candidate_dense)

                if similarity >= threshold:
                    filtered_candidates.append((similarity, text_content))

            if filtered_candidates:
                filtered_candidates.sort(key=lambda x: x[0], reverse=True)
                top_sim = filtered_candidates[0][0]

                coherent_group = [text for sim, text in filtered_candidates if (top_sim - sim) < 0.25]
                return coherent_group[:2]

            threshold *= 0.6

        best_match = raw_matches[0]
        payload = getattr(best_match, "payload", {})
        fallback_text = payload.get("text", str(best_match.key)) if isinstance(payload, dict) else str(best_match.key)
        return [fallback_text]

    def _decode_latent_to_text(
        self,
        cognitive_vector: torch.Tensor,
        prompt_hdc: Optional[torch.Tensor] = None,
        top_k: int = 10,
    ) -> str:
        if prompt_hdc is None:
            prompt_hdc = self.last_hdc

        hdc_target = self._expand_dense_to_hdc(_pad_or_trim(cognitive_vector, self.dense_dim))
        raw_matches = self.hdc.query(hdc_target, topk=top_k)

        if not raw_matches:
            if self.graph.nodes:
                top_node = max(self.graph.nodes.values(), key=lambda n: float(n.salience))
                return f"Concepto activo: {top_node.label}"
            return ""

        coherent_concepts = self._recurrent_refinement_loop(
            prompt_vector=prompt_hdc,
            raw_matches=raw_matches,
            max_iterations=3,
            initial_threshold=0.20,
        )

        return "\n".join(coherent_concepts)

    def reason(self, text: str, latent: torch.Tensor, hdc_vec: torch.Tensor) -> BrainDecision:
        self.statistics["reason_calls"] += 1

        # 1. Proceso de integración en LNN
        lnn_in = self._latent_to_lnn_input(latent, hdc_vec)
        lnn_out = self.lnn.step(lnn_in)
        self.last_lnn_out = lnn_out.clone()
        self.statistics["lnn_steps"] += 1

        # 2. Atractores y Grafo Cognitivo
        forcing_field = _normalize(_pad_or_trim(lnn_out, self.cfg.attractor_dim))
        stabilized_state = self.creativity.evolve(forcing_field, steps=5)

        mem_matches = self.hdc.query(hdc_vec, topk=5)

        max_sim = 0.0
        if mem_matches and hasattr(mem_matches[0], "hypervector"):
            max_sim = _cosine_similarity(
                self._compress_hdc_to_dense(hdc_vec),
                self._compress_hdc_to_dense(mem_matches[0].hypervector),
            )

        # 3. GATING DINÁMICO: ¿El sistema decide hablar?
        must_speak = self._should_generate_speech(lnn_out, latent, max_sim)

        answer_text = ""
        mode = "silent_cognition"

        if must_speak:
            answer_text = self._decode_latent_to_text(
                cognitive_vector=stabilized_state,
                prompt_hdc=hdc_vec,
                top_k=5,
            )
            mode = "expressive_cognition"

        confidence = float(
            torch.clamp(
                torch.linalg.vector_norm(stabilized_state) / 10.0,
                0.1,
                0.99,
            ).item()
        )

        return BrainDecision(
            actions=torch.tanh(stabilized_state[:128]),
            cognitive_state=stabilized_state,
            answer=answer_text,
            mode=mode,
            confidence=confidence,
            signals={
                "speech_gating_triggered": must_speak,
                "lnn_energy": float(torch.mean(lnn_out).item()),
                "attractor_norm": float(torch.linalg.vector_norm(stabilized_state).item()),
            },
        )

    def think(self, text: str, image: Any = None, sequence: Any = None, in_loop: Optional[bool] = False) -> str:
        with self._lock:
            self.turn_counter += 1
            self.statistics["turns"] = self.turn_counter
    
            perceptual_decision = self.observe(text=text, image=image, sequence=sequence)
    
            latent = self.last_latent
            hdc_vec = self._latent_to_hdc(latent, text=text)
            self.last_hdc = hdc_vec.clone()
    
            self.remember_observation(text, latent, hdc_vec)
            self.update_graph_from_text(text, latent)
    
            decision = self.reason(text, latent, hdc_vec)
    
            decision.actions = perceptual_decision.actions
            decision.cognitive_state = perceptual_decision.cognitive_state
            decision.attention_focus = perceptual_decision.attention_focus
    
            if self.turn_counter % self.cfg.auto_consolidate_every == 0:
                self.consolidate()
    
            if not in_loop:
                self.history.append(
                    {
                        "turn": self.turn_counter,
                        "text": text,
                        "answer": decision.answer,
                        "mode": decision.mode,
                        "confidence": decision.confidence,
                        "signals": decision.signals,
                        "actions": decision.actions.detach().cpu().tolist(),
                        "attention_focus": decision.attention_focus,
                    }
                )
                if len(self.history) > self.cfg.history_limit:
                    self.history = self.history[-self.cfg.history_limit :]
    
            if decision.confidence > 0.65:
                self.base.neuromodulators.boost("dopamina", 0.01)
                self.base.neuromodulators.boost("confianza", 0.01)
            else:
                self.base.neuromodulators.boost("curiosidad", 0.005)
    
            self.base.neuromodulators.clamp()
    
            self.lnn.neuromodulators = self.base.neuromodulators
            self.creativity.neuromodulators = self.base.neuromodulators
    
            optimizer = torch.optim.Adam([self.W_downproject, self.W_upproject], lr=1e-4)
            recon = F.linear(F.linear(hdc_vec, self.W_downproject), self.W_upproject.t())
            loss = F.mse_loss(recon, hdc_vec)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
    
            if not decision.answer:
                return "[Cerebro: Estímulo procesado internamente / Estado de silencio cognitivo]"
    
            return decision.answer

    def consolidate(self) -> None:
        recent_turns = self.history[-8:]
        turn_payloads = [{"role": "user", "text": t["text"]} for t in recent_turns]
        self.hdc_fusion.consolidate_from_turns(turn_payloads)

        if self.last_latent is not None and self.last_latent.numel() > 0:
            self.creativity.consolidate(
                _pad_or_trim(self.last_latent, self.cfg.attractor_dim),
                label=f"consolidated_{self.turn_counter}",
                energy=0.8,
            )

        self.hdc.cleanup(min_weight=0.10, min_access=0)
        self.base.neuromodulators.serotonina = min(1.0, self.base.neuromodulators.serotonina + 0.01)
        self.base.neuromodulators.clamp()

    def status(self) -> str:
        nm = self.base.neuromodulators
        return (
            f"BrainCore (Device: {self.device.type.upper()})\n"
            f"- turns: {self.turn_counter}\n"
            f"- memory: {len(self.hdc)}\n"
            f"- graph nodes: {len(self.graph.nodes)}\n"
            f"- graph edges: {len(self.graph.edges)}\n"
            f"- attractors: {len(self.creativity.attractors)}\n"
            f"- lnn state norm: {float(torch.linalg.vector_norm(self.lnn.get_state()).item()):.4f}\n"
            f"- dopamine: {nm.dopamina:.2f}\n"
            f"- serotonin: {nm.serotonina:.2f}\n"
            f"- noradrenaline: {nm.noradrenalina:.2f}\n"
            f"- acetylcholine: {nm.acetilcolina:.2f}\n"
            f"- adrenaline: {nm.adrenalina:.2f}\n"
            f"- curiosity: {nm.curiosidad:.2f}\n"
            f"- fatigue: {nm.fatiga:.2f}\n"
            f"- confidence: {nm.confianza:.2f}"
        )

    def explain_last(self) -> str:
        if not self.history:
            return "Sin historial."
        last = self.history[-1]
        return (
            f"turn={last['turn']} mode={last['mode']} conf={last['confidence']:.3f}\n"
            f"user={last['text']}\n"
            f"answer={last['answer']}\n"
            f"signals={last['signals']}"
        )

    def export(self) -> None:
        """
        Serialización binaria de ultra-alta velocidad guardando buffers tensoriales directamente.
        """
        return {
            "cfg": self.cfg,
            "turn_counter": self.turn_counter,
            "last_latent": self.last_latent.detach().cpu().tolist(),
            "last_hdc": self.last_hdc.detach().cpu().tolist(),
            "last_lnn_out": self.last_lnn_out.detach().cpu().tolist(),
            "history": self.history,
            "statistics": self.statistics,
            "lnn_state": self.lnn.export(),
            "attractor_state": self.creativity.export(),
            "hdc_state": self.hdc.export(),
        }

    def import_(self, payload: Dict[str, Any]) -> None:
        self.turn_counter = payload.get("turn_counter", 0)
        self.last_latent = torch.tensor(payload["last_latent"], dtype=DTYPE, device=self.device)
        self.last_hdc = torch.tensor(payload["last_hdc"], dtype=DTYPE, device=self.device)
        self.last_lnn_out = torch.tensor(payload["last_lnn_out"], dtype=DTYPE, device=self.device)
        self.history = payload.get("history", [])
        self.statistics = payload.get("statistics", self.statistics)

        if "lnn_state" in payload:
            self.lnn.import_(payload["lnn_state"])
        if "attractor_state" in payload:
            self.creativity.import_(payload["attractor_state"])
        if "hdc_state" in payload:
            self.hdc.import_(payload["hdc_state"])

    def save(self, filename: str = "braincore.pkl") -> Path:
        path = Path(filename)
        with path.open("wb") as f:
            pickle.dump(self.export(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load(self, filename: str = "braincore.pkl") -> Path:
        path = Path(filename)
        with path.open("rb") as f:
            payload = pickle.load(f)
        self.import_(payload)
        return path

    def train_on_file(self, filename: str) -> int:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return 0

        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

        n = 0
        for chunk in chunks:
            n += 1
            self.think(chunk)
            if n % 50 == 0:
                print(f"Bloque procesado: {n}")

        return len(chunks)


if __name__ == "__main__":
    brain = BrainCore()
    print(brain.think("El objeto rojo choca con el muro"))
    print(brain.status())