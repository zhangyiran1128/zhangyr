"""Lightweight MIDI event-grid autoencoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AutoencoderLoss:
    total: torch.Tensor
    note_bce: torch.Tensor
    duration_l1: torch.Tensor
    velocity_l1: torch.Tensor


class MidiAutoencoder(nn.Module):
    """Compress `R x L x S x P x F` event grids to `R x L x dz` latents."""

    def __init__(
        self,
        roles: int = 6,
        subdivisions: int = 16,
        pitches: int = 88,
        feature_dim: int = 3,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.roles = roles
        self.subdivisions = subdivisions
        self.pitches = pitches
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        flat_dim = subdivisions * pitches * feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode event grid tensor `(B, R, L, S, P, F)`."""

        b, r, l, s, p, f = x.shape
        if (r, s, p, f) != (self.roles, self.subdivisions, self.pitches, self.feature_dim):
            raise ValueError(f"Unexpected grid shape: {(r, s, p, f)}")
        flat = x.reshape(b * r * l, s * p * f)
        z = self.encoder(flat)
        return z.reshape(b, r, l, self.latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to event predictions.

        Channel 0 is note-on logits; channels 1 and 2 are transformed to positive
        duration and velocity in `[0, 1]`.
        """

        b, r, l, dz = z.shape
        flat = self.decoder(z.reshape(b * r * l, dz))
        raw = flat.reshape(b, r, l, self.subdivisions, self.pitches, self.feature_dim)
        note_logits = raw[..., 0:1]
        duration = F.softplus(raw[..., 1:2])
        velocity = torch.sigmoid(raw[..., 2:3])
        return torch.cat([note_logits, duration, velocity], dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def loss(self, recon: torch.Tensor, target: torch.Tensor) -> AutoencoderLoss:
        note_target = target[..., 0]
        note_logits = recon[..., 0]
        note_bce = F.binary_cross_entropy_with_logits(note_logits, note_target)
        active = (note_target > 0.5).float()
        denom = active.sum().clamp_min(1.0)
        duration_l1 = (F.smooth_l1_loss(recon[..., 1], target[..., 1], reduction="none") * active).sum() / denom
        velocity_l1 = (F.l1_loss(recon[..., 2], target[..., 2], reduction="none") * active).sum() / denom
        return AutoencoderLoss(note_bce + duration_l1 + velocity_l1, note_bce, duration_l1, velocity_l1)


def binarize_reconstruction(recon: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert decoder output to event-grid style values."""

    out = recon.clone()
    out[..., 0] = (torch.sigmoid(recon[..., 0]) >= threshold).float()
    out[..., 1] = torch.clamp(torch.round(recon[..., 1]), min=1)
    out[..., 2] = torch.clamp(recon[..., 2], 0.0, 1.0)
    out[..., 1:] *= out[..., :1]
    return out



def build_autoencoder(config: dict) -> MidiAutoencoder:
    data = config.get("data", {})
    settings = config.get("autoencoder", {})
    return MidiAutoencoder(
        roles=len(data.get("roles", ("lead", "bass", "rhythm", "harmony_pad", "accompaniment", "drums"))),
        subdivisions=int(data.get("subdivisions", 16)),
        pitches=int(data.get("pitch_max", 108)) - int(data.get("pitch_min", 21)) + 1,
        latent_dim=int(settings.get("latent_dim", 512)),
        hidden_dim=int(settings.get("hidden_dim", 1024)),
    )
