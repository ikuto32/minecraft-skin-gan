from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image

from src.config import EvalConfig
import torch.nn as nn
from tqdm import tqdm


@dataclass(frozen=True)
class EvalMetricsResult:
    fid: float | None = None
    kid_mean: float | None = None
    kid_std: float | None = None
    precision: float | None = None
    recall: float | None = None


def to_uint8_rgb(batch: torch.Tensor, *, metric_color_mode: str) -> torch.Tensor:
    """Convert normalized [0, 1] tensor images to uint8 for metric extractors.

    FID/KID backends expect RGB-like inputs, so alpha is intentionally dropped when
    `metric_color_mode="RGB"` and a 4ch batch is given.
    When `metric_color_mode="RGBA"`, keep 4 channels as-is.
    """
    if metric_color_mode == "RGB" and batch.size(1) == 4:
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
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device) if cfg.enable_fid else None
    kid = KernelInceptionDistance(
        subset_size=cfg.kid_subset_size, subsets=cfg.kid_subsets, feature=2048, normalize=False
    ).to(device) if cfg.enable_kid else None

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
            if real_seen < 2 and (cfg.enable_fid or cfg.enable_kid or cfg.enable_precision_recall):
                raise ValueError(
                    "Need at least 2 real samples to compute enabled metrics"
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
            if fake_seen < 2 and (cfg.enable_fid or cfg.enable_kid or cfg.enable_precision_recall):
                raise ValueError(
                    "Need at least 2 generated samples to compute enabled metrics"
                )
            fid_value: float | None = None
            kid_mean_value: float | None = None
            kid_std_value: float | None = None
            precision_value: float | None = None
            recall_value: float | None = None
            if fid is not None:
                _validate_fid_sample_counts(fid)
                fid_value = _compute_fid_value(fid)
            if kid is not None:
                kid_mean, kid_std = kid.compute()
                kid_mean_value = float(kid_mean.item())
                kid_std_value = float(kid_std.item())
            if cfg.enable_precision_recall:
                with TemporaryDirectory(prefix="eval_fake_samples_") as temp_dir:
                    fake_dir = Path(temp_dir)
                    _export_generated_samples(
                        cfg=cfg,
                        loader=loader,
                        generator=generator,
                        output_dir=fake_dir,
                        sample_count=cfg.sample_count,
                        device=device,
                    )
                    pr_metrics = calculate_metrics(
                        input1=str(cfg.real_dir),
                        input2=str(fake_dir),
                        cuda=device == "cuda",
                        isc=False,
                        fid=False,
                        kid=False,
                        prc=True,
                        verbose=False,
                    )
                    precision_value = float(pr_metrics["precision"])
                    recall_value = float(pr_metrics["recall"])

            return EvalMetricsResult(
                fid=fid_value,
                kid_mean=kid_mean_value,
                kid_std=kid_std_value,
                precision=precision_value,
                recall=recall_value,
            )

    finally:
        generator.train(was_training)


def _update_real_metrics(
    *,
    loader: DataLoader,
    fid: FrechetInceptionDistance | None,
    kid: KernelInceptionDistance | None,
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

            real_u8 = to_uint8_rgb(real, metric_color_mode=cfg.metric_color_mode)
            if fid is not None:
                fid.update(real_u8, real=True)
            if kid is not None:
                kid.update(real_u8, real=True)

            seen += take
            progress.update(take)

    return seen


def _update_fake_metrics(
    *,
    cfg: EvalConfig,
    loader: DataLoader,
    generator: nn.Module,
    fid: FrechetInceptionDistance | None,
    kid: KernelInceptionDistance | None,
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

            fake_u8 = to_uint8_rgb(fake, metric_color_mode=cfg.metric_color_mode)
            if fid is not None:
                fid.update(fake_u8, real=False)
            if kid is not None:
                kid.update(fake_u8, real=False)

            seen += take
            progress.update(take)

    return seen


def _export_generated_samples(
    *,
    cfg: EvalConfig,
    loader: DataLoader,
    generator: nn.Module,
    output_dir: Path,
    sample_count: int,
    device: str,
) -> int:
    seen = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for real in loader:
        if seen >= sample_count:
            break

        take = min(sample_count - seen, real.size(0))
        noise = torch.randn(take, cfg.z_dim, 1, 1, device=device)
        fake = generate_fake_batch(generator, noise).clamp(0, 1).cpu()

        for index_in_batch in range(take):
            image_path = output_dir / f"fake_{seen + index_in_batch:07d}.png"
            save_image(fake[index_in_batch], image_path)

        seen += take

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
