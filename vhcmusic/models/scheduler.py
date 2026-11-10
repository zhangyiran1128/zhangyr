"""DDPM training and DDIM sampling scheduler."""

from __future__ import annotations

import math

import torch


class DiffusionScheduler:
    """Noise scheduler for latent diffusion."""

    def __init__(self, steps: int = 1000, beta_schedule: str = "cosine", device: torch.device | str = "cpu"):
        self.steps = int(steps)
        self.device = torch.device(device)
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.steps, device=self.device)
        elif beta_schedule == "cosine":
            betas = self._cosine_betas(self.steps).to(self.device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        self.betas = betas.clamp(1e-5, 0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cat((torch.ones(1, device=self.device), self.alphas.cumprod(dim=0)))

    @staticmethod
    def _cosine_betas(steps: int, s: float = 0.008) -> torch.Tensor:
        x = torch.linspace(0, steps, steps + 1)
        alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas

    def to(self, device: torch.device | str) -> "DiffusionScheduler":
        return DiffusionScheduler(self.steps, "linear", device)._replace_from(self)

    def _replace_from(self, other: "DiffusionScheduler") -> "DiffusionScheduler":
        self.betas = other.betas.to(self.device)
        self.alphas = other.alphas.to(self.device)
        self.alpha_bars = other.alpha_bars.to(self.device)
        return self

    def q_sample(self, z0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(z0) if noise is None else noise
        a = self.alpha_bars.to(z0.device)[timesteps].view(-1, 1, 1, 1)
        zt = torch.sqrt(a) * z0 + torch.sqrt(1.0 - a) * noise
        return zt, noise

    def predict_x0(self, zt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bars.to(zt.device)[t].view(-1, 1, 1, 1)
        return (zt - torch.sqrt(1.0 - a) * eps) / torch.sqrt(a).clamp_min(1e-8)

    def ddim_timesteps(self, sampling_steps: int, start: int | None = None) -> torch.Tensor:
        start = self.steps if start is None else start
        if not 1 <= start <= self.steps or sampling_steps < 1:
            raise ValueError("Invalid DDIM start or step count")
        return torch.linspace(start, 1, min(sampling_steps, start), device=self.device).round().long()

    def ddim_step(self, zt: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eps: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        alpha_t = self.alpha_bars.to(zt.device)[t].view(-1, 1, 1, 1)
        alpha_prev = self.alpha_bars.to(zt.device)[t_prev].view(-1, 1, 1, 1)
        x0 = (zt - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t).clamp_min(1e-8)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t).clamp_min(1e-8) * (1 - alpha_t / alpha_prev.clamp_min(1e-8))).clamp_min(0)
        direction = torch.sqrt((1 - alpha_prev - sigma**2).clamp_min(0)) * eps
        noise = sigma * torch.randn_like(zt) if eta > 0 else 0.0
        return torch.sqrt(alpha_prev) * x0 + direction + noise
