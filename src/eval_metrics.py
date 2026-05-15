from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from src.config import EvalConfig
from src.models import Generator


@dataclass(frozen=True)
class EvalMetricsResult:
    fid: float
    kid_mean: float
    kid_std: float


def to_uint8_rgb(batch: torch.Tensor) -> torch.Tensor:
    if batch.size(1) == 4:
        batch = batch[:, :3]
    return (batch.clamp(0, 1) * 255).round().to(torch.uint8)


def generate_fake_batch(generator: Generator, noise: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fake = generator(noise)
    return (fake + 1.0) / 2.0


def compute_metrics(
    *,
    cfg: EvalConfig,
    loader: DataLoader,
    generator: Generator,
    device: str,
) -> EvalMetricsResult:
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        subset_size=cfg.kid_subset_size,
        subsets=cfg.kid_subsets,
        feature=2048,
        normalize=False,
    ).to(device)

    seen = 0
    for real in loader:
        if seen >= cfg.sample_count:
            break
        real = real.to(device)
        take = min(cfg.sample_count - seen, real.size(0))
        real = real[:take]

        noise = torch.randn(take, cfg.z_dim, 1, 1, device=device)
        fake = generate_fake_batch(generator, noise)

        real_u8 = to_uint8_rgb(real)
        fake_u8 = to_uint8_rgb(fake)

        fid.update(real_u8, real=True)
        fid.update(fake_u8, real=False)
        kid.update(real_u8, real=True)
        kid.update(fake_u8, real=False)
        seen += take

    fid_value = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    return EvalMetricsResult(
        fid=fid_value,
        kid_mean=float(kid_mean.item()),
        kid_std=float(kid_std.item()),
    )
