#!/usr/bin/env python
"""Fit the independent Section Analyzer and freeze its development-set peak threshold."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vhcmusic.metrics.control_metrics import section_boundary_f1
from vhcmusic.representation.midi_io import midi_to_event_grid
from vhcmusic.utils.checkpoint import save_checkpoint
from vhcmusic.utils.seed import seed_everything
from vhcmusic.verification.section_analyzer import LABELS, SECTION_CHECKPOINT, SectionPredictor, boundary_peaks, section_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", default=SECTION_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.width < 1:
        parser.error("epochs and width must be positive")
    seed_everything(args.seed)
    raw = args.manifest.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    splits = {split: [] for split in ("train", "val", "test")}
    provenance = {split: set() for split in splits}
    assigned = {}
    midi_splits = {}
    for record in records:
        if record["dataset"] != "SALAMI":
            raise ValueError("Section Analyzer training accepts only aligned SALAMI recordings")
        song, split = str(record["song_id"]), record["split"]
        midi = (args.manifest.parent / record["midi"]).resolve(strict=True)
        if split not in splits or assigned.setdefault(song, split) != split or midi_splits.setdefault(midi, split) != split:
            raise ValueError(f"SALAMI split overlap for {song}")
        provenance[split].add(song)
        if split == "test":
            continue
        x = section_features(midi_to_event_grid(midi))
        labels = torch.tensor([LABELS.index(label) for label in record["labels"]])
        if len(labels) != len(x):
            raise ValueError(f"Label count does not match MIDI bars for {song}")
        boundary = torch.zeros(len(x))
        for bar in record["boundaries"]:
            if not 1 <= bar <= len(x):
                raise ValueError(f"Boundary outside MIDI for {song}: {bar}")
            boundary[bar - 1] = 1
        splits[split].append((torch.from_numpy(x), boundary, labels))
    train, dev = splits["train"], splits["val"]
    if not train or not dev:
        raise ValueError("Distinct SALAMI training and development recordings are required")
    data = torch.cat([x for x, _, _ in train])
    mean, std = data.mean(0), data.std(0, unbiased=False).clamp_min(1e-4)
    device = torch.device(args.device)
    model = SectionPredictor(len(mean), args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    positives = sum(float(y[1:].sum()) for _, y, _ in train)
    negatives = sum(len(y) - 1 for _, y, _ in train) - positives
    weight = torch.tensor(max(1.0, negatives / max(1.0, positives)), device=device)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index in torch.randperm(len(train)).tolist():
            x, y, labels = train[index]
            boundary, function = model(((x - mean) / std)[None].to(device))
            loss = F.cross_entropy(function[0], labels.to(device))
            if len(y) > 1:
                loss = loss + F.binary_cross_entropy_with_logits(boundary[0, 1:], y[1:].to(device), pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            results = []
            for x, y, labels in dev:
                boundary, function = model(((x - mean) / std)[None].to(device))
                results.append(((boundary.cpu(), function.cpu()), y, labels))
        predictions = [(logit[0].sigmoid().numpy(), (y[1:].nonzero().flatten() + 2).tolist())
                       for (logit, _), y, _ in results]
        label_acc = np.mean([float((function[0].argmax(-1) == labels).float().mean())
                             for (_, function), _, labels in results])
        candidates = [(float(np.mean([section_boundary_f1(ref, boundary_peaks(prob, th, gap)[1:], 0)["f1"]
                                     for prob, ref in predictions])), th, gap)
                      for th in np.linspace(0.05, 0.95, 19) for gap in (1, 2, 4)]
        f1, threshold, gap = max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5), -item[2]))
        score = (f1 + label_acc) / 2
        if score > best:
            best = score
            save_checkpoint(args.output, dataset="SALAMI", manifest_sha256=hashlib.sha256(raw).hexdigest(),
                            splits={split: sorted(ids) for split, ids in provenance.items()}, labels=list(LABELS),
                            width=args.width, mean=mean.tolist(), std=std.tolist(), threshold=float(threshold),
                            min_gap=gap, epoch=epoch, seed=args.seed, development_f1=f1,
                            development_function_accuracy=float(label_acc), model_state=model.state_dict())
        print(f"epoch={epoch} development_boundary_f1={f1:.4f} development_function_accuracy={label_acc:.4f}")


if __name__ == "__main__":
    main()
