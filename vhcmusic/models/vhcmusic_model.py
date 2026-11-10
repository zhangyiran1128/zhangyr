"""Top-level VHCMusic latent diffusion model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vhcmusic.planning.schema import plan_bars

from .condition_encoder import ConditionEncoder
from .diffusion_unet import DiffusionDenoiser
from .midi_autoencoder import MidiAutoencoder, binarize_reconstruction, build_autoencoder
from .scheduler import DiffusionScheduler


class VHCMusicModel(nn.Module):
    """Composition of MIDI autoencoder, condition encoder, and latent denoiser."""

    def __init__(
        self,
        autoencoder: MidiAutoencoder,
        condition_encoder: ConditionEncoder,
        denoiser: DiffusionDenoiser,
        scheduler: DiffusionScheduler,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.condition_encoder = condition_encoder
        self.denoiser = denoiser
        self.scheduler = scheduler

    def freeze_autoencoder(self) -> None:
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()

    def forward(self, grid, plans, conditional_dropout=0.0, valid_bars=None):
        return self.diffusion_loss(grid, plans, conditional_dropout, valid_bars)

    def diffusion_loss(self, grid: torch.Tensor, plans: list[dict[str, Any]], conditional_dropout: float = 0.0, valid_bars=None) -> dict[str, torch.Tensor]:
        with torch.no_grad() if not any(p.requires_grad for p in self.autoencoder.parameters()) else torch.enable_grad():
            z0 = self.autoencoder.encode(grid)
        b = z0.shape[0]
        t = torch.randint(1, self.scheduler.steps + 1, (b,), device=z0.device)
        zt, noise = self.scheduler.q_sample(z0, t)
        drop_mask = None
        if conditional_dropout > 0:
            drop_mask = (torch.rand(b, device=z0.device) < conditional_dropout).float()
        cond = self.condition_encoder(plans, bars=z0.shape[2], device=z0.device, drop_mask=drop_mask)
        pred = self.denoiser(zt, t, cond, valid_bars)
        if valid_bars is None:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred.transpose(1, 2)[valid_bars], noise.transpose(1, 2)[valid_bars])
        return {"loss": loss, "mse": loss.detach()}

    @torch.no_grad()
    def sample_latents(
        self,
        plans: list[dict[str, Any]],
        bars: int,
        guidance_scale: float = 1.5,
        sampling_steps: int = 100,
        eta: float = 0.0,
        shape: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if any(plan_bars(plan) != bars for plan in plans):
            raise ValueError('Generation plans must cover exactly the requested bars')
        b = len(plans)
        r = self.autoencoder.roles
        dz = self.autoencoder.latent_dim
        shape = shape or (b, r, bars, dz)
        z = torch.randn(shape, device=device)
        cond = self.condition_encoder(plans, bars=bars, device=device)
        uncond = self.condition_encoder.null.expand_as(cond)
        timesteps = self.scheduler.ddim_timesteps(sampling_steps).to(device)
        previous = torch.cat((timesteps[1:], timesteps.new_zeros(1)))
        for t_scalar, prev_scalar in zip(timesteps, previous):
            t = t_scalar.expand(b)
            eps_cond = self.denoiser(z, t, cond)
            if guidance_scale != 1.0:
                eps_uncond = self.denoiser(z, t, uncond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond
            t_prev = prev_scalar.expand(b)
            z = self.scheduler.ddim_step(z, t, t_prev, eps, eta=eta)
        return z

    @torch.no_grad()
    def generate_grid(self, plans: list[dict[str, Any]], bars: int, **sample_kwargs) -> torch.Tensor:
        z = self.sample_latents(plans, bars=bars, **sample_kwargs)
        recon = self.autoencoder.decode(z)
        return binarize_reconstruction(recon)

    @torch.no_grad()
    def masked_edit_latents(
        self,
        z_original: torch.Tensor,
        plans: list[dict[str, Any]],
        edit_mask: torch.Tensor,
        edit_strength: float = 0.5,
        guidance_scale: float = 1.5,
        sampling_steps: int = 50,
    ) -> torch.Tensor:
        """DDIM editing with `M * generated + (1-M) * original_noisy` blending."""

        device = z_original.device
        b = z_original.shape[0]
        if not 0 <= edit_strength <= 1:
            raise ValueError("edit_strength must be between 0 and 1")
        edit_mask = torch.broadcast_to(edit_mask.to(device).bool(), z_original.shape[:3])[..., None]
        start = int(edit_strength * self.scheduler.steps)
        if start == 0 or not edit_mask.any():
            return z_original
        if len(plans) != b or any(plan_bars(plan) != z_original.shape[2] for plan in plans):
            raise ValueError('Edit plans must cover the input MIDI bars')
        cond = self.condition_encoder(plans, bars=z_original.shape[2], device=device)
        uncond = self.condition_encoder.null.expand_as(cond)
        timesteps = self.scheduler.ddim_timesteps(sampling_steps, start=start).to(device)
        z, fixed_noise = self.scheduler.q_sample(z_original, timesteps[0].repeat(b))
        previous = torch.cat((timesteps[1:], timesteps.new_zeros(1)))
        for t_scalar, prev_scalar in zip(timesteps, previous):
            t = t_scalar.expand(b)
            eps_cond = self.denoiser(z, t, cond)
            eps_uncond = self.denoiser(z, t, uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            t_prev = prev_scalar.expand(b)
            z_new = self.scheduler.ddim_step(z, t, t_prev, eps, eta=0.0)
            z_ori_prev, _ = self.scheduler.q_sample(z_original, t_prev, fixed_noise)
            z = torch.where(edit_mask, z_new, z_ori_prev)
        return z


def build_vhcmusic_model(config: dict[str, Any], device: torch.device | str | None = None) -> VHCMusicModel:
    """Instantiate VHCMusic modules from a training config dictionary."""

    data_cfg = config.get("data", {})
    diff_cfg = config.get("diffusion", {})
    roles = data_cfg.get("roles", ["lead", "bass", "rhythm", "harmony_pad", "accompaniment", "drums"])
    autoencoder = build_autoencoder(config)
    latent_dim = autoencoder.latent_dim
    model_dim = int(diff_cfg.get("model_dim", 512))
    condition_encoder = ConditionEncoder(dim=model_dim, roles=tuple(roles), mode=diff_cfg.get('condition_mode', 'full'))
    denoiser = DiffusionDenoiser(
        latent_dim=latent_dim,
        condition_dim=model_dim,
        model_dim=model_dim,
        num_layers=int(diff_cfg.get("num_layers", 6)),
        num_heads=int(diff_cfg.get("num_heads", 8)),
        ff_dim=int(diff_cfg.get("ff_dim", 2048)),
        dropout=float(diff_cfg.get("dropout", 0.1)),
        cross_track_attention=bool(diff_cfg.get("cross_track_attention", True)),
        max_bars=int(data_cfg.get("max_bars", 64)),
        roles=len(roles),
    )
    scheduler = DiffusionScheduler(
        steps=int(diff_cfg.get("steps", 1000)),
        beta_schedule=str(diff_cfg.get("beta_schedule", "cosine")),
        device=device or "cpu",
    )
    model = VHCMusicModel(autoencoder, condition_encoder, denoiser, scheduler)
    if device is not None:
        model.to(device)
    return model
