"""Masked latent sampling helpers."""

from __future__ import annotations

from typing import Any

import torch

from .edit_mask import EditMask


@torch.no_grad()
def masked_latent_edit(
    model,
    original_grid: torch.Tensor,
    plans: list[dict[str, Any]],
    edit_mask: EditMask,
    edit_strength: float = 0.5,
    guidance_scale: float = 1.5,
    sampling_steps: int = 50,
) -> torch.Tensor:
    """Encode original grid, run masked DDIM editing, and decode edited grid."""

    if not 0 <= edit_strength <= 1:
        raise ValueError("edit_strength must be between 0 and 1")
    if int(edit_strength * model.scheduler.steps) == 0 or not edit_mask.mask.any():
        return original_grid.clone()
    z_original = model.autoencoder.encode(original_grid)
    mask = edit_mask.to_tensor(device=original_grid.device).expand(original_grid.shape[0], -1, -1)
    z_edited = model.masked_edit_latents(
        z_original,
        plans,
        mask,
        edit_strength=edit_strength,
        guidance_scale=guidance_scale,
        sampling_steps=sampling_steps,
    )
    from vhcmusic.models.midi_autoencoder import binarize_reconstruction

    return binarize_reconstruction(model.autoencoder.decode(z_edited))



def edit_grid(model, original, plan, mask, strength, guidance, steps):
    from dataclasses import replace

    device = next(model.parameters()).device
    tensor = torch.from_numpy(original.data[None]).to(device)
    edited = masked_latent_edit(model, tensor, [plan], mask, edit_strength=strength,
                                guidance_scale=guidance, sampling_steps=steps)
    return replace(original, data=edited[0].cpu().numpy())
