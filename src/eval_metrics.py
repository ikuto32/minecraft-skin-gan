from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from src.config import EvalConfig
import torch.nn as nn


@dataclass(frozen=True)
class EvalMetricsResult:
    fid: float
    kid_mean: float
    kid_std: float


def to_uint8_rgb(batch: torch.Tensor) -> torch.Tensor:
    if batch.size(1) == 4:
        batch = batch[:, :3]
    return (batch.clamp(0, 1) * 255).round().to(torch.uint8)


def generate_fake_batch(generator: nn.Module, noise: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        fake = generator(noise)
    return (fake + 1.0) / 2.0


def _metric_sample_count(metric: object, *, real: bool) -> int | None:
    attr_name = "real_features_num_samples" if real else "fake_features_num_samples"
    raw = getattr(metric, attr_name, None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        return int(raw.item())
    return int(raw)


def compute_metrics(
    *,
    cfg: EvalConfig,
    loader: DataLoader,
    generator: nn.Module,
    device: str,
    real_features_state: dict[str, object] | None,
    cache_real_only: bool,
) -> EvalMetricsResult | dict[str, object]:
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        subset_size=cfg.kid_subset_size,
        subsets=cfg.kid_subsets,
        feature=2048,
        normalize=False,
    ).to(device)

    if real_features_state is not None:
        fid.load_state_dict(real_features_state["fid_state"])
        kid.load_state_dict(real_features_state["kid_state"])
    else:
        seen = 0
        for real in loader:
            if seen >= cfg.sample_count:
                break
            real = real.to(device)
            take = min(cfg.sample_count - seen, real.size(0))
            real = real[:take]
            real_u8 = to_uint8_rgb(real)
            fid.update(real_u8, real=True)
            kid.update(real_u8, real=True)
            seen += take
        if seen < 2:
            raise ValueError("Need at least 2 real samples to compute FID/KID metrics")

    if cache_real_only:
        return {
            "fid_state": fid.state_dict(),
            "kid_state": kid.state_dict(),
        }

    seen = 0
    for real in loader:
        if seen >= cfg.sample_count:
            break
        take = min(cfg.sample_count - seen, real.size(0))
        noise = torch.randn(take, cfg.z_dim, 1, 1, device=device)
        fake = generate_fake_batch(generator, noise)
        fake_u8 = to_uint8_rgb(fake)
        fid.update(fake_u8, real=False)
        kid.update(fake_u8, real=False)
        seen += take
    if seen < 2:
        raise ValueError("Need at least 2 generated samples to compute FID/KID metrics")

    real_count = _metric_sample_count(fid, real=True)
    fake_count = _metric_sample_count(fid, real=False)
    if real_count is not None and real_count < 2:
        raise ValueError(
            "Need at least 2 real samples to compute FID. "
            f"Got {real_count}; clear cached real features and increase eval.sample_count."
        )
    if fake_count is not None and fake_count < 2:
        raise ValueError(
            "Need at least 2 generated samples to compute FID. "
            f"Got {fake_count}; increase eval.sample_count."
        )

    try:
        fid_value = float(fid.compute().item())
    except RuntimeError as exc:
        message = str(exc)
        if "More than one sample is required" in message:
            raise ValueError(
                "FID requires at least 2 real and 2 generated samples. "
                "Try increasing eval.sample_count and deleting cached real features."
            ) from exc
        raise
    kid_mean, kid_std = kid.compute()
    return EvalMetricsResult(
        fid=fid_value,
        kid_mean=float(kid_mean.item()),
        kid_std=float(kid_std.item()),
    )
