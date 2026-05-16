from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from src.config import EvalConfig
import torch.nn as nn
from tqdm import tqdm


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
) -> EvalMetricsResult:
    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False,
    ).to(device)

    kid = KernelInceptionDistance(
        subset_size=cfg.kid_subset_size,
        subsets=cfg.kid_subsets,
        feature=2048,
        normalize=False,
    ).to(device)

    was_training = generator.training
    generator.eval()

    try:
        with torch.inference_mode():
            real_seen = _update_real_metrics(
                loader=loader,
                fid=fid,
                kid=kid,
                sample_count=cfg.sample_count,
                device=device,
            )
            if real_seen < 2:
                raise ValueError(
                    "Need at least 2 real samples to compute FID/KID metrics"
                )

            fake_seen = _update_fake_metrics(
                cfg=cfg,
                loader=loader,
                generator=generator,
                fid=fid,
                kid=kid,
                sample_count=cfg.sample_count,
                device=device,
            )
            if fake_seen < 2:
                raise ValueError(
                    "Need at least 2 generated samples to compute FID/KID metrics"
                )

            _validate_fid_sample_counts(fid)

            fid_value = _compute_fid_value(fid)
            kid_mean, kid_std = kid.compute()

            return EvalMetricsResult(
                fid=fid_value,
                kid_mean=float(kid_mean.item()),
                kid_std=float(kid_std.item()),
            )

    finally:
        generator.train(was_training)


def _update_real_metrics(
    *,
    loader: DataLoader,
    fid: FrechetInceptionDistance,
    kid: KernelInceptionDistance,
    sample_count: int,
    device: str,
) -> int:
    seen = 0

    with tqdm(
        total=sample_count,
        desc="Updating real samples",
        unit="img",
        leave=False,
    ) as progress:
        for real in loader:
            if seen >= sample_count:
                break

            take = min(sample_count - seen, real.size(0))
            real = real[:take].to(device, non_blocking=True)

            real_u8 = to_uint8_rgb(real)
            fid.update(real_u8, real=True)
            kid.update(real_u8, real=True)

            seen += take
            progress.update(take)

    return seen


def _update_fake_metrics(
    *,
    cfg: EvalConfig,
    loader: DataLoader,
    generator: nn.Module,
    fid: FrechetInceptionDistance,
    kid: KernelInceptionDistance,
    sample_count: int,
    device: str,
) -> int:
    seen = 0

    with tqdm(
        total=sample_count,
        desc="Updating generated samples",
        unit="img",
        leave=False,
    ) as progress:
        for real in loader:
            if seen >= sample_count:
                break

            take = min(sample_count - seen, real.size(0))

            noise = torch.randn(
                take,
                cfg.z_dim,
                1,
                1,
                device=device,
            )
            fake = generate_fake_batch(generator, noise)

            fake_u8 = to_uint8_rgb(fake)
            fid.update(fake_u8, real=False)
            kid.update(fake_u8, real=False)

            seen += take
            progress.update(take)

    return seen


def _validate_fid_sample_counts(
    fid: FrechetInceptionDistance,
) -> None:
    real_count = _metric_sample_count(fid, real=True)
    fake_count = _metric_sample_count(fid, real=False)

    if real_count is not None and real_count < 2:
        raise ValueError(
            "Need at least 2 real samples to compute FID. "
            f"Got {real_count}; increase eval.sample_count."
        )

    if fake_count is not None and fake_count < 2:
        raise ValueError(
            "Need at least 2 generated samples to compute FID. "
            f"Got {fake_count}; increase eval.sample_count."
        )


def _compute_fid_value(
    fid: FrechetInceptionDistance,
) -> float:
    try:
        return float(fid.compute().item())
    except RuntimeError as exc:
        if "More than one sample is required" in str(exc):
            raise ValueError(
                "FID requires at least 2 real and 2 generated samples. "
                "Try increasing eval.sample_count."
            ) from exc
        raise