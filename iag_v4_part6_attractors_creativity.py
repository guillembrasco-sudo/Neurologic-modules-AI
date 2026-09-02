"""
IAG Modular V4 - Parte 6
Sistema de atractores y creatividad.

Objetivos:
- Explorar espacios latentes no lineales
- Generar soluciones divergentes
- Mantener atractores estables y mutables
- Modulación por neuromoduladores
- Integración con HDC / LNN / GNN
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from common import Neuromodulators, set_seed, DEVICE, DTYPE


# ============================================================
# Utilidades de Tensores
# ============================================================

def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(torch.clamp(x, -20.0, 20.0))


def _clip(x: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    return torch.clamp(x, -limit, limit)


# ============================================================
# Configuración y Neuromoduladores
# ============================================================

@dataclass
class AttractorConfig:
    latent_dim: int = 256
    attractor_count: int = 16
    seed: int = 7
    step_size: float = 0.05
    exploration_scale: float = 0.45
    stability_scale: float = 0.55
    novelty_threshold: float = 0.15
    max_steps: int = 64
    output_count: int = 4

# ============================================================
# Atractores y Candidatos Creativos
# ============================================================

@dataclass
class Attractor:
    center: torch.Tensor
    radius: float
    energy: float
    label: str
    metadata: Dict[str, Any]

    def compact(self) -> Dict[str, Any]:
        return {
            "center": self.center.detach().cpu().tolist(),
            "radius": float(self.radius),
            "energy": float(self.energy),
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class CreativeCandidate:
    vector: torch.Tensor
    score: float
    novelty: float
    stability: float
    label: str
    explanation: str


# ============================================================
# Motor de Atractores Latentes
# ============================================================

class AttractorEngine(nn.Module):
    def __init__(
        self,
        cfg: Optional[AttractorConfig] = None,
        neuromodulators: Optional[Neuromodulators] = None,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.cfg = cfg or AttractorConfig()
        self._target_device = device
        self.neuromodulators = neuromodulators or Neuromodulators()

        self.dim = self.cfg.latent_dim
        self.register_buffer("state", torch.zeros(self.dim, dtype=DTYPE, device=self._target_device))
        self.register_buffer("bias", torch.zeros(self.dim, dtype=DTYPE, device=self._target_device))
        self.register_buffer("W_rec", torch.zeros((self.dim, self.dim), dtype=DTYPE, device=self._target_device))

        self.noise_level = 0.05
        self.dt = self.cfg.step_size

        self.attractors: List[Attractor] = []
        self.history: deque[torch.Tensor] = deque(maxlen=1000)

        # Generador PyTorch determinista
        self.gen = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            self.gen.manual_seed(self.cfg.seed)

        self._init_default_attractors()
        self._update_recurrent_weights()

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return self._target_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dev = x.to(device=self.device, dtype=DTYPE)
        self.state.copy_(_normalize(x_dev))
        return self.state.clone()

    def _init_default_attractors(self) -> None:
        for i in range(self.cfg.attractor_count):
            center = self._random_center()
            radius = float(((1.25 - 0.35) * torch.rand(1, generator=self.gen) + 0.35).item())
            energy = float(((1.0 - 0.25) * torch.rand(1, generator=self.gen) + 0.25).item())
            label = f"attractor_{i}"
            self.attractors.append(
                Attractor(
                    center=center,
                    radius=radius,
                    energy=energy,
                    label=label,
                    metadata={"kind": "default"},
                )
            )

    def _random_center(self) -> torch.Tensor:
        v = torch.randn(self.cfg.latent_dim, generator=self.gen, dtype=DTYPE).to(self.device)
        return _normalize(v)

    def _update_recurrent_weights(self) -> None:
        if not self.attractors:
            return

        centers_tensor = torch.stack([a.center.to(device=self.device) for a in self.attractors])
        energies_tensor = torch.tensor([a.energy for a in self.attractors], device=self.device, dtype=DTYPE).unsqueeze(1)
        radii_tensor = torch.tensor([a.radius for a in self.attractors], device=self.device, dtype=DTYPE).unsqueeze(1)

        weighted_centers = centers_tensor * torch.sqrt(energies_tensor)
        W = torch.matmul(weighted_centers.T, weighted_centers)
        norm_W = torch.linalg.matrix_norm(W) + 1e-8
        self.W_rec.copy_(W / norm_W)
        
        spectral_norm = torch.linalg.matrix_norm(W, ord=2) + 1e-8
        self.W_rec.copy_((W / spectral_norm) * 0.55)

        self.register_buffer("centers", centers_tensor, persistent=False)
        self.register_buffer("radii", radii_tensor, persistent=False)
        self.register_buffer("energies", energies_tensor, persistent=False)

    def add_attractor(
        self,
        center: torch.Tensor,
        radius: float = 0.8,
        energy: float = 0.8,
        label: str = "custom",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(center, torch.Tensor):
            center_tensor = torch.tensor(center, dtype=DTYPE, device=self.device)
        else:
            center_tensor = center.to(device=self.device, dtype=DTYPE)

        self.attractors.append(
            Attractor(
                center=_normalize(center_tensor),
                radius=float(radius),
                energy=float(energy),
                label=label,
                metadata=metadata or {},
            )
        )

        self._update_recurrent_weights()

    def step_attractor_dynamics(self, steps: int = 1) -> torch.Tensor:
        nm = self.neuromodulators
        nm.clamp()
        
        exploration = self.cfg.exploration_scale * (1.0 + nm.curiosidad - nm.serotonina)
        noise_amp = self.noise_level * float(torch.clamp(torch.tensor(exploration), 0.1, 3.0).item())

        for _ in range(steps):
            diffs = self.centers - self.state # Broadcasting: (N, D) - (D,) -> (N, D)
            dists = torch.linalg.norm(diffs, dim=1, keepdim=True) + 1e-8

            mask = (dists < self.radii).float()

            magnitudes = (self.energies / dists) * mask
            force = torch.sum(magnitudes * diffs, dim=0)

            rec_influence = torch.matmul(self.W_rec, self.state)
            noise = torch.randn_like(self.state) * noise_amp
            
            ds = _tanh(force + rec_influence + self.bias) + noise
            self.state = _normalize(self.state + self.dt * ds)
            self.history.append(self.state.detach().clone())

        return self.state.clone()

    def generate_creative_candidates(self, count: Optional[int] = None) -> List[CreativeCandidate]:
        k = count or self.cfg.output_count
        candidates = []

        if len(self.history) > 0:
            past_reference = torch.stack(list(self.history), dim=0)
        else:
            past_reference = None
        
        for i in range(k):
            vec = self.step_attractor_dynamics(steps=self.cfg.max_steps)

            if past_reference is not None and past_reference.shape[0] > 0:
                sims = torch.matmul(past_reference, vec)
                novelty = float((1.0 - torch.max(sims)).item())
            else:
                novelty = 1.0

            stability = float(torch.linalg.vector_norm(torch.matmul(self.W_rec, vec)).item())
            score = 0.6 * novelty + 0.4 * stability

            candidates.append(
                CreativeCandidate(
                    vector=vec,
                    score=score,
                    novelty=novelty,
                    stability=stability,
                    label=f"candidate_{i}",
                    explanation=f"Cand {i}: Nov={novelty:.3f}, Stab={stability:.3f}"
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    # --------------------------------------------------------
    # Dinámica
    # --------------------------------------------------------

    def _modulated_params(self) -> Tuple[float, float, float]:
        nm = self.neuromodulators
        nm.clamp()

        exploration = float(self.cfg.exploration_scale * (1.0 + 0.45 * nm.curiosidad - 0.50 * nm.adrenalina - 0.25 * nm.fatiga))
        stability = float(self.cfg.stability_scale * (1.0 + 0.50 * nm.serotonina + 0.25 * nm.confianza - 0.10 * nm.adrenalina))
        novelty_thresh = float(self.cfg.novelty_threshold * (1.0 + 0.30 * nm.curiosidad - 0.20 * nm.confianza))
        return exploration, stability, novelty_thresh

    def energy(self, x: torch.Tensor, attractor: Attractor) -> float:
        x = _normalize(x)
        d = float(torch.linalg.vector_norm(x - attractor.center).item())
        # energía tipo cuenca: menor cuando está cerca del centro
        return d * d / (attractor.radius * attractor.radius + 1e-8) - attractor.energy

    def nearest_attractors(self, x: torch.Tensor, topk: int = 5) -> List[Tuple[Attractor, float]]:
        vals = [(a, self.energy(x, a)) for a in self.attractors]
        vals.sort(key=lambda t: t[1])
        return vals[: int(topk)]

    def step_towards(self, x: torch.Tensor, target: Attractor, step_size: Optional[float] = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=DTYPE, device=self.device)
        step = self.cfg.step_size if step_size is None else float(step_size)
        direction = target.center - x
        return _normalize(x + step * direction)

    def evolve(self, external_forcing: torch.Tensor, steps: int = 10, temperature: float = 1.0) -> torch.Tensor:
        """
        Evoluciona el estado latente del sistema dentro del paisaje de energía 
        hacia un punto fijo o ciclo límite.
        """
        if not isinstance(external_forcing, torch.Tensor):
            forcing = torch.tensor(external_forcing, dtype=DTYPE, device=self.device)
        else:
            forcing = external_forcing.to(device=self.device, dtype=DTYPE)

        forcing = _normalize(forcing)
        self.state = forcing.clone()

        for _ in range(steps):
            # Gradiente de energía potencial (sistema de Hopfield continuo)
            energy_gradient = -self.W_rec @ self.state + self.bias
            
            # Término estocástico de fluctuación térmica
            noise_std = self.noise_level * temperature
            noise = torch.randn_like(self.state) * noise_std
            
            # Proyección no lineal hiperbólica
            dx = (-self.state - energy_gradient + forcing) * self.dt + noise
            self.state = _tanh(self.state + dx)
            
            self.history.append(self.state.detach().clone())
            
        return self.state

    # --------------------------------------------------------
    # Generación Creativa
    # --------------------------------------------------------

    def generate_variants_batch(
        self,
        seed_vector: torch.Tensor,
        count: int = 4,
        diversity: float = 0.35,
        steps: int = 10
    ) -> torch.Tensor:
        # 1. Expandir semilla a forma en lote (N, D)
        if seed_vector.ndim == 1:
            seeds = seed_vector.unsqueeze(0).repeat(count, 1)
        else:
            seeds = seed_vector.repeat(count, 1)

        # 2. Inyección de ruido paralelo vectorizado
        noise = torch.randn_like(seeds) * diversity
        states = F.normalize(seeds + noise, p=2.0, dim=-1)

        # 3. Evolución paralela sobre la dimensión de Batch (N, D)
        for _ in range(steps):
            # Gradiente de energía vectorizado: (N, D) @ (D, D) -> (N, D)
            energy_grad = -torch.matmul(states, self.W_rec.t()) + self.bias
            dx = (-states - energy_grad) * self.dt
            states = torch.tanh(states + dx)

        return states

    def generate_variants(
        self,
        seed_vector: torch.Tensor,
        count: Optional[int] = None,
        diversity: float = 0.35,
    ) -> List[CreativeCandidate]:
        """
        Produce varias soluciones candidatas desde un punto semántico de partida.
        """
        if not isinstance(seed_vector, torch.Tensor):
            seed_vec = torch.tensor(seed_vector, dtype=DTYPE, device=self.device)
        else:
            seed_vec = seed_vector.to(device=self.device, dtype=DTYPE)

        seed_vec = _normalize(seed_vec)
        count = self.cfg.output_count if count is None else int(count)
        exploration, stability, creativity = self._modulated_params()

        candidates: List[CreativeCandidate] = []
        for i in range(count):
            x = seed_vec.clone()

            # Variante por perturbación + evolución
            perturb = torch.randn_like(x) * diversity
            perturb = _normalize(perturb)
            x = _normalize(x + diversity * perturb)

            steps = int(self.cfg.max_steps * (0.5 + 0.5 * creativity))
            temp_mod = max(0.25, 1.0 - 0.25 * exploration)
            evolved = self.evolve(x, steps=steps, temperature=temp_mod)

            ranked = self.nearest_attractors(evolved, topk=1)
            if ranked:
                attractor, energy_val = ranked[0]
                novelty = float(torch.linalg.vector_norm(evolved - seed_vec).item())
                stability_score = float(1.0 / (1.0 + max(0.0, energy_val)))
                score = float((0.45 * novelty) + (0.35 * stability_score) + (0.20 * creativity))
                explanation = (
                    f"Derivada hacia {attractor.label} con energía={energy_val:+.3f}, "
                    f"novelty={novelty:.3f}, stability={stability_score:.3f}"
                )
                candidates.append(
                    CreativeCandidate(
                        vector=evolved,
                        score=score,
                        novelty=novelty,
                        stability=stability_score,
                        label=attractor.label,
                        explanation=explanation,
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def brainstorm(
        self,
        context_vector: torch.Tensor,
        topk: int = 4,
        diversity: float = 0.40,
    ) -> List[CreativeCandidate]:
        """
        Generación de ideas divergentes en el espacio latente.
        """
        base = _normalize(context_vector)
        candidates = self.generate_variants(base, count=topk, diversity=diversity)

        # Modulación final: adrenalina reduce creatividad, dopamina la consolida
        nm = self.neuromodulators
        if nm.adrenalina > 0.75:
            for c in candidates:
                c.score *= 0.80
                c.explanation += " | alta adrenalina: se prioriza cierre"
        if nm.dopamina > 0.65:
            for c in candidates:
                c.score *= 1.10
                c.explanation += " | dopamina alta: se consolida"

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # --------------------------------------------------------
    # Aprendizaje de Atractores
    # --------------------------------------------------------

    def consolidate(self, vector: torch.Tensor, label: str = "learned", energy: float = 0.75) -> None:
        """
        Crea o ajusta un atractor nuevo a partir de una solución útil.
        """
        if not isinstance(vector, torch.Tensor):
            vec = torch.tensor(vector, dtype=DTYPE, device=self.device)
        else:
            vec = vector.to(device=self.device, dtype=DTYPE)

        vec = _normalize(vec)

        if not self.attractors:
            self.add_attractor(vec, radius=0.75, energy=energy, label=label, metadata={"kind": "consolidated"})
            return

        # Si existe uno muy cercano, ajustarlo en lugar de crear otro
        best_idx = None
        best_dist = float("inf")
        for i, a in enumerate(self.attractors):
            d = float(torch.linalg.vector_norm(vec - a.center).item())
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist < self.cfg.novelty_threshold:
            a = self.attractors[best_idx]
            a.center = _normalize(0.9 * a.center + 0.1 * vec)
            a.radius = float(0.95 * a.radius + 0.05 * 0.9)
            a.energy = float(0.95 * a.energy + 0.05 * energy)
            a.metadata["updated"] = True
        else:
            self.add_attractor(vec, radius=0.75, energy=energy, label=label, metadata={"kind": "consolidated"})
        
        self._update_recurrent_weights()

    def replay_memory(self, memory_vectors: List[torch.Tensor]) -> None:
        for i, v in enumerate(memory_vectors):
            self.consolidate(v, label=f"replay_{i}", energy=0.7)

    # --------------------------------------------------------
    # Diagnóstico
    # --------------------------------------------------------

    def summary(self) -> str:
        return (
            f"AttractorEngine (Device: {self.device.type.upper()})\n"
            f"- attractors: {len(self.attractors)}\n"
            f"- history: {len(self.history)}\n"
            f"- latent_dim: {self.cfg.latent_dim}\n"
            f"- output_count: {self.cfg.output_count}"
        )

    # --------------------------------------------------------
    # Persistencia
    # --------------------------------------------------------

    def export(self) -> Dict[str, Any]:
        return {
            "cfg": self.cfg,
            "attractors": [a.compact() for a in self.attractors],
            "history": [h.detach().cpu().tolist() for h in self.history],
            "nm": self.neuromodulators,
        }

    def import_(self, payload: Dict[str, Any]) -> None:
        self.cfg = payload.get("cfg", self.cfg)
        self.attractors = []
        for a in payload.get("attractors", []):
            center_tensor = _normalize(torch.tensor(a["center"], dtype=DTYPE, device=self.device))
            self.attractors.append(
                Attractor(
                    center=center_tensor,
                    radius=float(a.get("radius", 0.8)),
                    energy=float(a.get("energy", 0.8)),
                    label=str(a.get("label", "attractor")),
                    metadata=dict(a.get("metadata", {})),
                )
            )
        self.history = deque(
            [torch.tensor(v, dtype=DTYPE, device=self.device) for v in payload.get("history", [])],
            maxlen=1000,
        )
        self.neuromodulators = payload.get("nm", self.neuromodulators)
        self._update_recurrent_weights()


class AttractorStore:
    def save(self, engine: AttractorEngine, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(engine.export(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load(self, path: str | Path, device: torch.device = DEVICE) -> AttractorEngine:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        cfg = payload.get("cfg", AttractorConfig())
        engine = AttractorEngine(cfg=cfg, neuromodulators=payload.get("nm", Neuromodulators()), device=device)
        engine.import_(payload)
        return engine


if __name__ == "__main__":
    cfg = AttractorConfig(latent_dim=128, attractor_count=8)
    engine = AttractorEngine(cfg=cfg)

    seed = torch.randn(128, device=DEVICE, dtype=DTYPE)
    seed = _normalize(seed)

    ideas = engine.brainstorm(seed, topk=4, diversity=0.30)
    print(engine.summary())
    for i, c in enumerate(ideas, 1):
        print(i, c.label, f"{c.score:.3f}", f"nov={c.novelty:.3f}", c.explanation)