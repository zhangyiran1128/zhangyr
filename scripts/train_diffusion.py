#!/usr/bin/env python
"""Train the conditional epsilon denoiser with a fixed pretrained MIDI autoencoder."""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generator_data import GeneratorDataset, collate_grids, generator_items, grid_statistics
from vhcmusic.models.vhcmusic_model import build_vhcmusic_model
from vhcmusic.utils.checkpoint import link_checkpoint, load_checkpoint, save_checkpoint
from vhcmusic.utils.config import load_config
from vhcmusic.utils.seed import seed_everything


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for key, value in self.shadow.items():
            value.lerp_(state[key].detach(), 1 - self.decay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_vhcmusic.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data, diffusion = cfg["data"], cfg["diffusion"]
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = cfg["device"]["prefer_cuda"] and torch.cuda.is_available()
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if cuda and dist.is_nccl_available() else "gloo")
    if diffusion["batch_size"] % world:
        raise ValueError("Global batch size must be divisible by the number of workers")
    seed_everything(seed)
    out = Path(diffusion["checkpoint_dir"]) / f"seed_{seed}"
    items = generator_items(data["processed_dir"])
    statistics = grid_statistics(items) if rank == 0 else None
    if world > 1:
        shared = [statistics]
        dist.broadcast_object_list(shared, src=0)
        statistics = shared[0]
    train = GeneratorDataset(items, "train", statistics=statistics, max_bars=data["max_bars"])
    val = GeneratorDataset(items, "val", statistics=statistics, max_bars=data["max_bars"]) if rank == 0 else None
    sampler = DistributedSampler(train, shuffle=True, seed=seed) if world > 1 else None
    loader_args = {"batch_size": diffusion["batch_size"] // world, "collate_fn": collate_grids,
                   "num_workers": data["num_workers"], "persistent_workers": data["num_workers"] > 0}
    train_loader = DataLoader(train, sampler=sampler, shuffle=sampler is None, **loader_args)
    val_loader = DataLoader(val, **loader_args) if val is not None else None
    model = build_vhcmusic_model(cfg, device)
    resume = args.resume or diffusion.get("resume")
    ckpt = load_checkpoint(resume, map_location=device) if resume else None
    autoencoder = ckpt["autoencoder"] if ckpt else load_checkpoint(cfg["autoencoder"]["checkpoint"], map_location=device)["model_state"]
    model.autoencoder.load_state_dict(autoencoder)
    model.freeze_autoencoder()
    if ckpt:
        if ckpt["seed"] != seed or ckpt["statistics"] != statistics:
            raise ValueError("Resume requires the original training seed and data statistics")
        model.condition_encoder.load_state_dict(ckpt["condition_encoder"])
        model.denoiser.load_state_dict(ckpt["denoiser"])
    ema = EMA(model, diffusion["ema_decay"])
    ema.shadow = {key: value for key, value in ema.shadow.items() if not key.startswith("autoencoder.")}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=diffusion["lr"], weight_decay=diffusion["weight_decay"])
    amp = bool(diffusion["amp"] and cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if ckpt:
        ema.shadow = ckpt["ema"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
    start, best = (ckpt["epoch"] + 1, ckpt["best"]) if ckpt else (1, float("inf"))
    parallel = DistributedDataParallel(model, device_ids=[local] if cuda else None,
                                       find_unused_parameters=False) if world > 1 else model
    seed_everything(seed + rank)
    val_model = build_vhcmusic_model(cfg, device) if rank == 0 else None
    if val_model is not None:
        val_model.autoencoder.load_state_dict(autoencoder)
        val_model.freeze_autoencoder()
        val_model.eval()
    for epoch in range(start, diffusion["epochs"] + 1):
        seed_everything(seed + epoch * world + rank)
        if sampler is not None:
            sampler.set_epoch(epoch)
        parallel.train()
        model.autoencoder.eval()
        for grid, plans, valid in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = parallel(grid.to(device), plans, diffusion["conditional_dropout"], valid.to(device))["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, diffusion["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                ema.update(model)
        if rank == 0:
            val_model.load_state_dict({**val_model.state_dict(), **ema.shadow})
            total = tokens = 0
            with torch.inference_mode(), torch.random.fork_rng(devices=[local] if cuda else []):
                torch.manual_seed(seed)
                for grid, plans, valid in val_loader:
                    n = int(valid.sum())
                    val_loss = val_model.diffusion_loss(grid.to(device), plans, valid_bars=valid.to(device))["loss"]
                    total += float(val_loss) * n
                    tokens += n
            loss = total / tokens
            improved = loss < best
            best = min(best, loss)
            payload = {"autoencoder": model.autoencoder.state_dict(), "condition_encoder": model.condition_encoder.state_dict(),
                       "denoiser": model.denoiser.state_dict(), "ema": ema.shadow, "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict(), "config": cfg, "epoch": epoch, "seed": seed, "statistics": statistics,
                       "val_loss": loss, "best": best}
            save_checkpoint(out / "last.pt", **payload)
            if improved:
                link_checkpoint(out / "last.pt", out / "best.pt")
            print(f"epoch={epoch} val_epsilon_mse={loss:.6f}", flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
