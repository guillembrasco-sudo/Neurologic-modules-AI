"""
IAG Modular V4 - Parte 5
Liquid Neural Network (LNN) continua, vectorizada y neuromodulada.

Características:
- Dinámica de tiempo continuo dx/dt = f(x, t, u)
- Integración Euler / Heun / RK4
- Neuronas líquidas con constantes temporales heterogéneas
- Plasticidad online modulada por neuromoduladores
- Salida codificable a HDC
- Estabilidad numérica con clipping y normalización
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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

def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(torch.clamp(x, -20.0, 20.0))


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def _clip(x: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    return torch.clamp(x, -limit, limit)


# ============================================================
# Configuraciones y Estado Neuromodulador
# ============================================================

@dataclass
class LNNConfig:
    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 128
    seed: int = 7
    integration: str = "rk4"  # euler | heun | rk4
    dt: float = 0.05
    state_clip: float = 8.0
    output_clip: float = 8.0
    plasticity_enabled: bool = True
    plasticity_rate: float = 0.002
    plasticity_decay: float = 0.01
    recurrent_scale: float = 0.6
    input_scale: float = 0.8
    output_scale: float = 0.7


# ============================================================
# Red Neuronal Líquida (LNN)
# ============================================================

class LiquidNeuralNetwork(nn.Module):
    def __init__(
        self,
        cfg: Optional[LNNConfig] = None,
        neuromodulators: Optional[Neuromodulators] = None,
        device: Optional[torch.device] = DEVICE,
    ):
        super().__init__()
        self.cfg = cfg or LNNConfig()
        self._target_device = device or DEVICE
        self.neuromodulators = neuromodulators or Neuromodulators()

        h = self.cfg.hidden_dim
        d_in = self.cfg.input_dim
        d_out = self.cfg.output_dim

        self.plasticity_enabled = self.cfg.plasticity_enabled
        self.lr_plasticity = self.cfg.plasticity_rate
        self.tau_leak = self.cfg.plasticity_decay

        scale_in = self.cfg.input_scale / math.sqrt(max(1, d_in))
        scale_rec = self.cfg.recurrent_scale / math.sqrt(max(1, h))
        scale_out = self.cfg.output_scale / math.sqrt(max(1, h))

        # Inicializador reproducible mediante PyTorch Generator
        gen = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            gen.manual_seed(self.cfg.seed)

        self.W_in = nn.Parameter((torch.randn(h, d_in, generator=gen) * scale_in).to(dtype=DTYPE, device=self._target_device))
        self.W_rec = nn.Parameter((torch.randn(h, h, generator=gen) * scale_rec).to(dtype=DTYPE, device=self._target_device))
        self.W_out = nn.Parameter((torch.randn(d_out, h, generator=gen) * scale_out).to(dtype=DTYPE, device=self._target_device))
        self.tau = nn.Parameter(torch.ones(h, dtype=DTYPE, device=self._target_device) * 1.0)
        self.b = nn.Parameter(torch.zeros((h,), dtype=DTYPE))
        self.bo = nn.Parameter(torch.zeros((d_out,), dtype=DTYPE))

        self.register_buffer("tau_base", ((2.50 - 0.15) * torch.rand(h, generator=gen) + 0.15).to(dtype=DTYPE))
        self.register_buffer("gain", ((1.2 - 0.8) * torch.rand(h, generator=gen) + 0.8).to(dtype=DTYPE))

        self.register_buffer("state", torch.zeros((h,), dtype=DTYPE))
        self.register_buffer("trace", torch.zeros((h,), dtype=DTYPE))
        self.register_buffer("output_state", torch.zeros((d_out,), dtype=DTYPE))
        self.register_buffer("last_input", torch.zeros((d_in,), dtype=DTYPE))
        self.register_buffer("last_error", torch.zeros((d_out,), dtype=DTYPE))

        self.step_count = 0

        self.noise_scale = 0.01
        self.state_decay = 0.001
        self.trace_decay = 0.97

        self.target_energy_rec = float(torch.sum(torch.abs(self.W_rec)).item())
        self.target_energy_in = float(torch.sum(torch.abs(self.W_in)).item())

    @property
    def device(self) -> torch.device:
        return self.W_in.device

    def _ode_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Ecuación diferencial continua dx/dt = (-x + tanh(W_in @ u + W_rec @ x)) / tau"""
        pre_act = F.linear(u, self.W_in) + F.linear(x, self.W_rec)
        dxdt = (-x + torch.tanh(pre_act)) / torch.clamp(self.tau, min=0.1)
        return dxdt

    def _modulated_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nm = self.neuromodulators
        mod_rec = self.W_rec * (1.0 + 0.2 * (nm.dopamina - 0.5))
        mod_in = self.W_in * (1.0 + 0.2 * (nm.noradrenalina - 0.5))
        tau_eff = self.tau_base * (1.0 - 0.3 * nm.acetilcolina)
        return mod_in, mod_rec, tau_eff

    def dynamics(self, x: torch.Tensor, u: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        mod_in, mod_rec, tau = self._modulated_params()
        pre = F.linear(x, mod_rec) + F.linear(u, mod_in) + self.b
        drive = self.gain * _tanh(pre)

        nm = self.neuromodulators
        noise_amp = self.noise_scale * (1.0 + 0.6 * nm.adrenalina + 0.2 * nm.curiosidad - 0.3 * nm.serotonina)
        noise = torch.randn_like(x) * noise_amp

        dx = (-x + drive + noise) / (tau + 1e-6)
        dx = dx - self.state_decay * self.state_decay * x
        return _clip(dx, self.cfg.state_clip)

    def _euler_step(self, x: torch.Tensor, u: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        dx = self.dynamics(x, u, t)
        new_x = x + dt * dx
        return _clip(new_x, self.cfg.state_clip)

    def forward(self, u: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """Pase hacia adelante estándar unificado con la integración continua."""
        return self.step_continuous_adaptive(u, dt=dt)

    def _heun_step(self, x: torch.Tensor, u: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        k1 = self.dynamics(x, u, t)
        k2 = self.dynamics(x + dt * k1, u, t + dt)
        return x + 0.5 * dt * (k1 + k2)

    def _rk4_step(self, x: torch.Tensor, u: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        k1 = self.dynamics(x, u, t)
        k2 = self.dynamics(x + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self.dynamics(x + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self.dynamics(x + dt * k3, u, t + dt)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, u: torch.Tensor, dt: Optional[float] = None, t: Optional[float] = None) -> torch.Tensor:
        """Alias de compatibilidad que redirige a forward."""
        return self.forward(u, dt=dt)

    def _apply_homeostasis(self) -> None:
        """
        Fuerza disipativa metabólica que actúa como atractor homeostático.
        """
        nm = self.neuromodulators
        baselines = {
            "dopamina": 0.35,
            "serotonina": 0.35,
            "noradrenalina": 0.35,
            "acetilcolina": 0.35,
            "adrenalina": 0.05,
            "curiosidad": 0.50,
            "fatiga": 0.00,
            "confianza": 0.50,
        }
        metabolic_decay = 0.01

        for field_name, baseline_val in baselines.items():
            current_val = float(getattr(nm, field_name))
            stabilized_val = current_val + metabolic_decay * (baseline_val - current_val)
            setattr(nm, field_name, stabilized_val)

        nm.clamp()

    def _online_plasticity(self, post_act: torch.Tensor, u: torch.Tensor) -> None:
        """
        Aprendizaje Hebbiano local con modulación por compuerta biológica.
        post_act: [hidden_dim]
        u: [input_dim]
        """
        if not self.plasticity_enabled:
            return

        _, mod_rec_scale, _ = self._modulated_params()
        # acetilcolina module la tasa de plasticidad, no la matriz completa
        lr = self.lr_plasticity * (1.0 + 0.5 * (self.neuromodulators.acetilcolina - 0.5))

        with torch.no_grad():
            dw_in = lr * torch.outer(post_act, u) - (self.tau_leak * self.W_in.data)
            self.W_in.data.add_(dw_in)

            dw_rec = lr * torch.outer(post_act, self.state) - (self.tau_leak * self.W_rec.data)
            self.W_rec.data.add_(dw_rec)

            self._spectral_stabilize()
            self._apply_homeostatic_scaling()

    def _apply_homeostatic_scaling(self) -> None:
        """
        Reescala multiplicativamente las matrices de pesos para evitar saturación Hebbiana.
        """
        energy_rec = float(torch.sum(torch.abs(self.W_rec)).item())
        if energy_rec > 0:
            self.W_rec *= self.target_energy_rec / energy_rec

        energy_in = float(torch.sum(torch.abs(self.W_in)).item())
        if energy_in > 0:
            self.W_in *= self.target_energy_in / energy_in

    def _spectral_stabilize(self) -> None:
        """
        Acrotamiento espectral del peso recurrente para garantizar estabilidad y acotamiento.
        """
        with torch.no_grad():
            row_sums = torch.sum(torch.abs(self.W_rec), dim=1)
            max_domain = float(torch.max(row_sums).item())

            target_radius = 0.95
            if max_domain > target_radius:
                self.W_rec.mul_(target_radius / (max_domain + 1e-8))

    def reset_state(self) -> None:
        self.state.zero_()
        self.trace.zero_()
        self.output_state.zero_()
        self.last_input.zero_()
        self.last_error.zero_()

    def get_state(self) -> torch.Tensor:
        return self.state.clone()

    def get_output(self) -> torch.Tensor:
        return self.output_state.clone()

    def encode_output_as_hdc(self, dim: int = 10000, seed: int = 7) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        basis_a = torch.where(torch.rand(dim, generator=gen) > 0.5, 1.0, -1.0).to(device=self.device, dtype=DTYPE)
        basis_b = torch.where(torch.rand(dim, generator=gen) > 0.5, 1.0, -1.0).to(device=self.device, dtype=DTYPE)

        if self.output_state.numel() == 0:
            return basis_a

        value = float(torch.clamp(torch.mean(self.output_state) * 0.5 + 0.5, 0.0, 1.0).item())
        hv = (1.0 - value) * basis_a + value * basis_b
        return _normalize(torch.where(hv >= 0.0, 1.0, -1.0))

    def diagnostics(self) -> str:
        nm = self.neuromodulators
        return (
            f"LiquidNeuralNetwork (Device: {self.device.type.upper()})\n"
            f"- step_count: {self.step_count}\n"
            f"- input_dim: {self.cfg.input_dim}\n"
            f"- hidden_dim: {self.cfg.hidden_dim}\n"
            f"- output_dim: {self.cfg.output_dim}\n"
            f"- integration: {self.cfg.integration}\n"
            f"- state_norm: {float(torch.linalg.vector_norm(self.state).item()):.4f}\n"
            f"- output_norm: {float(torch.linalg.vector_norm(self.output_state).item()):.4f}\n"
            f"- dopamine: {nm.dopamina:.2f}\n"
            f"- serotonin: {nm.serotonina:.2f}\n"
            f"- noradrenaline: {nm.noradrenalina:.2f}\n"
            f"- acetylcholine: {nm.acetilcolina:.2f}\n"
            f"- adrenaline: {nm.adrenalina:.2f}\n"
            f"- curiosity: {nm.curiosidad:.2f}\n"
            f"- fatigue: {nm.fatiga:.2f}\n"
            f"- confidence: {nm.confianza:.2f}"
        )

    def export(self) -> Dict[str, Any]:
        return {
            "cfg": self.cfg,
            "W_in": self.W_in.detach().cpu().tolist(),
            "W_rec": self.W_rec.detach().cpu().tolist(),
            "W_out": self.W_out.detach().cpu().tolist(),
            "b": self.b.detach().cpu().tolist(),
            "bo": self.bo.detach().cpu().tolist(),
            "tau_base": self.tau_base.detach().cpu().tolist(),
            "gain": self.gain.detach().cpu().tolist(),
            "state": self.state.detach().cpu().tolist(),
            "trace": self.trace.detach().cpu().tolist(),
            "output_state": self.output_state.detach().cpu().tolist(),
            "last_input": self.last_input.detach().cpu().tolist(),
            "last_error": self.last_error.detach().cpu().tolist(),
            "step_count": self.step_count,
            "nm": self.neuromodulators,
        }

    def import_(self, payload: Dict[str, Any]) -> None:
        self.cfg = payload.get("cfg", self.cfg)
        with torch.no_grad():
            self.W_in.data.copy_(torch.tensor(payload["W_in"], dtype=DTYPE, device=self.device))
            self.W_rec.data.copy_(torch.tensor(payload["W_rec"], dtype=DTYPE, device=self.device))
            self.W_out.data.copy_(torch.tensor(payload["W_out"], dtype=DTYPE, device=self.device))
            self.b.data.copy_(torch.tensor(payload["b"], dtype=DTYPE, device=self.device))
            self.bo.data.copy_(torch.tensor(payload["bo"], dtype=DTYPE, device=self.device))
        self.tau_base = torch.tensor(payload["tau_base"], dtype=DTYPE, device=self.device)
        self.gain = torch.tensor(payload["gain"], dtype=DTYPE, device=self.device)
        self.state = torch.tensor(payload["state"], dtype=DTYPE, device=self.device)
        self.trace = torch.tensor(payload["trace"], dtype=DTYPE, device=self.device)
        self.output_state = torch.tensor(payload["output_state"], dtype=DTYPE, device=self.device)
        self.last_input = torch.tensor(payload["last_input"], dtype=DTYPE, device=self.device)
        self.last_error = torch.tensor(payload["last_error"], dtype=DTYPE, device=self.device)
        self.step_count = int(payload.get("step_count", 0))
        self.neuromodulators = payload.get("nm", self.neuromodulators)

    def step_continuous_adaptive(self, u: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """
        Avanza la LNN utilizando un paso de integración adaptativo continuo sobre PyTorch.
        """
        self.step_count += 1
        step_dt = float(self.cfg.dt if dt is None else dt)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=DTYPE, device=self.device)
        else:
            u = u.to(device=self.device, dtype=DTYPE)

        u = u.reshape(-1)
        if u.numel() < self.cfg.input_dim:
            u = F.pad(u, (0, self.cfg.input_dim - u.numel()))
        elif u.numel() > self.cfg.input_dim:
            u = u[: self.cfg.input_dim]

        self.last_input.copy_(u)
        t = float(self.step_count * step_dt)

        if self.cfg.integration == "euler":
            new_state = self._euler_step(self.state, u, t, step_dt)
        elif self.cfg.integration == "heun":
            new_state = self._heun_step(self.state, u, t, step_dt)
        else:
            new_state = self._rk4_step(self.state, u, t, step_dt)

        self.state.copy_(_clip(new_state, self.cfg.state_clip))

        act = _tanh(self.state)
        self.trace.copy_(self.trace_decay * self.trace + (1.0 - self.trace_decay) * act)

        out = self.W_out @ self.state + self.bo
        out = _clip(out, self.cfg.output_clip)
        self.output_state.copy_(_tanh(out))

        self._online_plasticity(act, u)
        self._apply_homeostasis()

        return self.output_state.clone()


class LNNStore(nn.Module):
    def save(self, net: LiquidNeuralNetwork, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(net.export(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load(self, path: str | Path, device: torch.device = DEVICE) -> LiquidNeuralNetwork:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        cfg = payload.get("cfg", LNNConfig())
        net = LiquidNeuralNetwork(cfg=cfg, neuromodulators=payload.get("nm", Neuromodulators()), device=device)
        net.import_(payload)
        return net


if __name__ == "__main__":
    cfg = LNNConfig(input_dim=256, hidden_dim=192, output_dim=128, integration="rk4")
    net = LiquidNeuralNetwork(cfg=cfg)

    x = torch.randn(256, device=DEVICE, dtype=DTYPE)
    for _ in range(10):
        y = net.step(x)

    print(net.diagnostics())
    print("Salida shape:", y.shape)
    print("HDC shape:", net.encode_output_as_hdc(dim=2048).shape)