#!/usr/bin/env python
"""Render MIDI to Music.wav using FluidSynth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vhcmusic.rendering.fluidsynth_render import render_midi_to_wav
from vhcmusic.utils.config import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/render.yaml")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--output", default="outputs/render")
    args = parser.parse_args()
    cfg = load_config(args.config)["render"]
    out_dir = ensure_dir(args.output)
    name = cfg.pop("final_name")
    wav = render_midi_to_wav(args.midi, out_dir / name, **cfg)
    print(wav)


if __name__ == "__main__":
    main()
