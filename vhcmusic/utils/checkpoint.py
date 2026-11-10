"""Checkpoint helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def save_checkpoint(path: str | Path, **payload: Any) -> Path:
    """Save a PyTorch checkpoint."""

    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, suffix=".pt.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint with a clear error message."""

    import torch

    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return torch.load(ckpt, map_location=map_location, weights_only=True)



def link_checkpoint(source: str | Path, destination: str | Path) -> Path:
    out = Path(destination)
    with NamedTemporaryFile(dir=out.parent, suffix=".pt.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return out
