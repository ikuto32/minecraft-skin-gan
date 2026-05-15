from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, epoch: int, generator, discriminator, opt_g, opt_d, *, generator_ema=None, best_metric: float | None = None):
    payload = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }
    if generator_ema is not None:
        payload["generator_ema"] = generator_ema.state_dict()
    if best_metric is not None:
        payload["best_metric"] = best_metric
    torch.save(payload, path)


def load_checkpoint(path: Path, generator, discriminator, opt_g, opt_d, device: str, *, generator_ema=None):
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    if generator_ema is not None and "generator_ema" in ckpt:
        generator_ema.load_state_dict(ckpt["generator_ema"])
    return ckpt["epoch"] + 1, float(ckpt.get("best_metric", float("inf")))
