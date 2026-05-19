from __future__ import annotations

from pathlib import Path
import subprocess

import torch

CURRENT_CHECKPOINT_SCHEMA_VERSION = 3


def _summarize_train_config(train_config) -> dict[str, object]:
    if train_config is None:
        return {}

    keys = (
        "epochs",
        "batch_size",
        "z_dim",
        "lr",
        "seed",
        "r1_gamma",
        "r1_interval",
        "r2_gamma",
        "r2_interval",
        "performance_profile",
        "amp_dtype",
        "ema_beta",
        "use_ada",
        "ada_target",
        "ada_interval",
        "ada_speed",
        "best_metric",
    )
    return {k: getattr(train_config, k) for k in keys if hasattr(train_config, k)}


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def save_checkpoint(
    path: Path,
    epoch: int,
    generator,
    discriminator,
    opt_g,
    opt_d,
    *,
    generator_ema=None,
    best_metric: float | None = None,
    train_config=None,
    model_name: str = "unknown",
    model_hparams: dict[str, object] | None = None,
    ada_p: float | None = None,
):
    payload = {
        "schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "model_name": model_name,
        "model_hparams": model_hparams or {},
        "model_type": {
            "generator": generator.__class__.__name__,
            "discriminator": discriminator.__class__.__name__,
        },
        "train_config": _summarize_train_config(train_config),
        "git_commit": _get_git_commit(),
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
    if ada_p is not None:
        payload["ada_p"] = float(ada_p)
    torch.save(payload, path)


def load_checkpoint(path: Path, generator, discriminator, opt_g, opt_d, device: str, *, generator_ema=None, expected_model_name: str | None = None, expected_model_hparams: dict[str, object] | None = None):
    ckpt = torch.load(path, map_location=device)
    schema_version = int(ckpt.get("schema_version", 0))
    if schema_version != CURRENT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible checkpoint schema_version={schema_version} (expected {CURRENT_CHECKPOINT_SCHEMA_VERSION})"
        )

    actual_model_name = ckpt.get("model_name")
    actual_model_hparams = ckpt.get("model_hparams", {})
    if expected_model_name is not None and actual_model_name != expected_model_name:
        raise ValueError(f"Checkpoint model_name mismatch: {actual_model_name!r} != {expected_model_name!r}")
    if expected_model_hparams is not None and actual_model_hparams != expected_model_hparams:
        raise ValueError("Checkpoint model_hparams mismatch")

    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    if generator_ema is not None and "generator_ema" in ckpt:
        generator_ema.load_state_dict(ckpt["generator_ema"])
    return {
        "start_epoch": ckpt["epoch"] + 1,
        "best_metric": float(ckpt.get("best_metric", float("inf"))),
        "schema_version": schema_version,
        "model_type": ckpt.get("model_type"),
        "model_name": actual_model_name,
        "model_hparams": actual_model_hparams,
        "ada_p": float(ckpt.get("ada_p", 0.0)),
    }
