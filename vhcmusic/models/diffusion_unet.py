"""Conditional denoiser for VHCMusic latent diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn

from .cross_track_attention import CrossTrackAttention


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device).float() / max(1, half - 1))
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class DenoiserBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, ff_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, valid_bars=None) -> torch.Tensor:
        b, r, l, d = h.shape
        seq = self.norm1(h).reshape(b * r, l, d)
        padding = None if valid_bars is None else (~valid_bars).repeat_interleave(r, dim=0)
        y, _ = self.self_attn(seq, seq, seq, key_padding_mask=padding, need_weights=False)
        h = h + self.dropout(y.reshape(b, r, l, d))
        h = h + self.dropout(self.ff(self.norm2(h)))
        return h


class DiffusionDenoiser(nn.Module):
    """Transformer-style latent denoiser with optional cross-track attention."""

    def __init__(
        self,
        latent_dim: int = 512,
        condition_dim: int = 512,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        cross_track_attention: bool = True,
        max_bars: int = 64,
        roles: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Identity() if latent_dim == model_dim else nn.Linear(latent_dim, model_dim)
        self.cond_proj = nn.Identity() if condition_dim == model_dim else nn.Linear(condition_dim, model_dim)
        self.time_emb = SinusoidalTimeEmbedding(model_dim)
        self.bar_emb = nn.Embedding(max_bars, model_dim)
        self.role_emb = nn.Embedding(roles, model_dim)
        self.blocks = nn.ModuleList([DenoiserBlock(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)])
        self.cross_track = CrossTrackAttention(model_dim, num_heads, dropout) if cross_track_attention and roles > 1 else None
        if self.cross_track is not None and num_layers < 3:
            raise ValueError('Cross-track attention requires at least three blocks')
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor, valid_bars=None) -> torch.Tensor:
        b, r, l, _ = z_t.shape
        h = self.input_proj(z_t) + self.cond_proj(cond)
        h = h + self.time_emb(timesteps)[:, None, None, :]
        if not 1 <= l <= self.bar_emb.num_embeddings or r != self.role_emb.num_embeddings:
            raise ValueError(f'Invalid latent layout: {r} roles and {l} bars')
        bar_ids = torch.arange(l, device=z_t.device)
        role_ids = torch.arange(r, device=z_t.device)
        h = h + self.bar_emb(bar_ids)[None, None, :, :] + self.role_emb(role_ids)[None, :, None, :]
        for index, block in enumerate(self.blocks, 1):
            h = block(h, valid_bars)
            if index == 3 and self.cross_track is not None:
                h = self.cross_track(h)
        return self.out_proj(self.out_norm(h))
