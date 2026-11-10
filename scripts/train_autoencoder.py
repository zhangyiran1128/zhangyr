#!/usr/bin/env python
"""Train the shared track-bar MIDI autoencoder using generator training songs."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generator_data import GeneratorDataset, collate_grids, generator_items
from vhcmusic.models.midi_autoencoder import build_autoencoder
from vhcmusic.utils.checkpoint import save_checkpoint
from vhcmusic.utils.config import load_config
from vhcmusic.utils.seed import seed_everything


def epoch_loss(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total = tokens = 0
    with torch.set_grad_enabled(optimizer is not None):
        for grid, _, valid in loader:
            # Treat the valid track-bar tokens independently, as in Section 3.3.
            target = grid.transpose(1, 2)[valid].unsqueeze(2).to(device)
            reconstruction, _ = model(target)
            loss = model.loss(reconstruction, target).total
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(target)
            tokens += len(target)
    return total / tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_vhcmusic.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data, ae = cfg["data"], cfg["autoencoder"]
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    seed_everything(seed)
    device = torch.device("cuda" if cfg["device"]["prefer_cuda"] and torch.cuda.is_available() else "cpu")
    items = generator_items(data["processed_dir"])
    loaders = [DataLoader(GeneratorDataset(items, split, max_bars=data["max_bars"]),
                          batch_size=ae["batch_size"], shuffle=split == "train",
                          num_workers=data["num_workers"], persistent_workers=data["num_workers"] > 0,
                          collate_fn=collate_grids)
               for split in ("train", "val")]
    model = build_autoencoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ae["lr"], weight_decay=ae.get("weight_decay", 0.01))
    best = float("inf")
    for epoch in range(1, ae["epochs"] + 1):
        train_loss = epoch_loss(model, loaders[0], device, optimizer)
        val_loss = epoch_loss(model, loaders[1], device)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            save_checkpoint(ae["checkpoint"], model_state=model.state_dict(), config=cfg,
                            epoch=epoch, val_loss=val_loss, seed=seed)


if __name__ == "__main__":
    main()
