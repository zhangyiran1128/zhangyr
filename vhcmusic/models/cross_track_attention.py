"""Same-bar multi-track attention for VHCMusic."""

from __future__ import annotations

import torch
from torch import nn


class CrossTrackAttention(nn.Module):
    """Exchange information among functional tracks within each bar.

    Input shape is `(B, R, L, D)`. For every bar `l`, roles are treated as the
    attention sequence so lead, bass, rhythm, pad, accompaniment, and drums can
    coordinate while preserving bar positions.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, r, l, d = h.shape
        x = self.norm(h).permute(0, 2, 1, 3).reshape(b * l, r, d)
        mask = torch.eye(r, dtype=torch.bool, device=h.device)
        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        y = y.reshape(b, l, r, d).permute(0, 2, 1, 3)
        return h + self.dropout(y)
