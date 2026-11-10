#!/usr/bin/env python
"""Generate MIDI from a fixed plan or a schema-constrained GPT-4o plan."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vhcmusic.models.vhcmusic_model import build_vhcmusic_model
from vhcmusic.planning.music_planner import MusicPlanner, normalize_plan
from vhcmusic.planning.schema import plan_bars, validate_plan
from vhcmusic.rendering.fluidsynth_render import render_midi_to_wav
from vhcmusic.representation.event_grid import EventGrid
from vhcmusic.representation.midi_io import write_event_grid_midi
from vhcmusic.utils.checkpoint import load_checkpoint
from vhcmusic.utils.config import ensure_dir, load_config
from vhcmusic.utils.seed import seed_everything
from vhcmusic.verification.chord_analyzer import ChordAnalyzer
from vhcmusic.verification.section_analyzer import SectionAnalyzer
from vhcmusic.verification.verification_report import build_verification_report


def load_plan(args, roles):
    if args.plan:
        plan = normalize_plan(json.loads(Path(args.plan).read_text(encoding="utf-8-sig")))
        validate_plan(plan, roles=roles)
        if args.bars is not None and args.bars != plan_bars(plan):
            raise ValueError("--bars must equal the number of bars in the fixed plan")
        return plan
    payload = json.loads(Path(args.prompt).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and payload.get("keyframes"):
        base = Path(args.prompt).resolve().parent
        payload["keyframes"] = [dict(frame, path=str(base / frame["path"])) if isinstance(frame, dict)
                                else str(base / frame) for frame in payload["keyframes"]]
    bars = args.bars or (int(payload.get("bars", 32)) if isinstance(payload, dict) else 32)
    return MusicPlanner(roles=roles).plan(payload, bars=bars)


def restore_model(config, checkpoint, device, use_ema=True):
    ckpt = load_checkpoint(checkpoint, map_location=device)
    missing = {"autoencoder", "condition_encoder", "denoiser", "statistics"} - ckpt.keys()
    if missing:
        raise ValueError(f"Expected a trained VHCMusic diffusion checkpoint. Missing keys: {sorted(missing)}")
    saved = ckpt["config"]
    for section, names in {
        "data": ("max_bars", "roles", "subdivisions", "pitch_min", "pitch_max"),
        "autoencoder": ("latent_dim", "hidden_dim"),
        "diffusion": ("model_dim", "num_layers", "num_heads", "ff_dim", "steps", "beta_schedule", "cross_track_attention"),
    }.items():
        if any(saved[section][key] != config[section][key] for key in names):
            raise ValueError(f"Checkpoint and configured {section} architecture differ")
    if saved["diffusion"].get("condition_mode", "full") != config["diffusion"].get("condition_mode", "full"):
        raise ValueError("Checkpoint conditioning mode differs from the configuration")
    model = build_vhcmusic_model(config, device=device)
    model.autoencoder.load_state_dict(ckpt["autoencoder"])
    model.condition_encoder.load_state_dict(ckpt["condition_encoder"])
    model.denoiser.load_state_dict(ckpt["denoiser"])
    if use_ema:
        state = model.state_dict()
        expected = {key for key in state if not key.startswith("autoencoder.")}
        if set(ckpt["ema"]) != expected:
            raise ValueError("Checkpoint EMA does not cover the condition encoder and denoiser")
        model.load_state_dict({**state, **ckpt["ema"]})
    model.statistics = ckpt["statistics"]
    return model.eval()


def generate_grid(model, plan, config):
    bars = plan_bars(plan)
    data, diffusion = config["data"], config["diffusion"]
    if not 1 <= bars <= data["max_bars"]:
        raise ValueError(f"Plan must contain 1–{data['max_bars']} bars")
    # The paper specifies the target tempo as a MIDI control and uses a fixed bar grid.
    tempo = float(plan.get("tempo", 120.0))
    if not 0 < tempo <= 1000:
        raise ValueError("MIDI tempo must be positive and at most 1000 BPM")
    notes = model.generate_grid([plan], bars=bars, guidance_scale=diffusion["classifier_free_guidance"],
                                sampling_steps=diffusion["sampling_steps"])[0].cpu().numpy()
    return EventGrid(notes, roles=tuple(data["roles"]), subdivisions=data["subdivisions"],
                     pitch_min=data["pitch_min"], pitch_max=data["pitch_max"], tempo=tempo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_vhcmusic.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--plan")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bars", type=int)
    parser.add_argument("--output", default="outputs/generated")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--render-config", default="configs/render.yaml")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--section-checkpoint", help="Write online observations using this SALAMI checkpoint")
    parser.add_argument("--chord-calibration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    out = ensure_dir(args.output)
    plan = load_plan(args, tuple(cfg["data"]["roles"]))
    (out / "hierarchical_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    device = torch.device("cuda" if cfg["device"]["prefer_cuda"] and torch.cuda.is_available() else "cpu")
    model = restore_model(cfg, args.checkpoint, device, use_ema=not args.no_ema)
    grid = generate_grid(model, plan, cfg)
    midi = write_event_grid_midi(grid, out / "generated.mid")
    if args.section_checkpoint:
        report = build_verification_report(plan, grid, section_analyzer=SectionAnalyzer(args.section_checkpoint),
                                           chord_analyzer=ChordAnalyzer.from_calibration(args.chord_calibration),
                                           density_scale=model.statistics["density_scale"])
        (out / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_render:
        render = load_config(args.render_config)["render"].copy()
        name = render.pop("final_name")
        render_midi_to_wav(midi, out / name, duration_seconds=grid.bars * 4 * 60 / grid.tempo, **render)
    print(midi)


if __name__ == "__main__":
    main()
