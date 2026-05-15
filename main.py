from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import TrainConfig
from src.engine import train
from src.eval import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minecraft skin GAN training/evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DCGAN model with Hydra overrides")
    train_parser.add_argument("overrides", nargs="*", help="Hydra style overrides (e.g., epochs=200 tracking.backend=wandb)")

    eval_parser = subparsers.add_parser("eval", help="Run FID/KID evaluation from a checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--real-dir", default="data/skins")
    eval_parser.add_argument("--output-dir", default="outputs/eval")
    eval_parser.add_argument("--sample-count", type=int, default=2048)
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--z-dim", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=1234)
    eval_parser.add_argument("--num-workers", type=int, default=2)
    eval_parser.add_argument("--kid-subsets", type=int, default=50)
    eval_parser.add_argument("--kid-subset-size", type=int, default=32)
    eval_parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    eval_parser.add_argument("--epoch", type=int, default=None)
    return parser


def _build_train_config(overrides: list[str]) -> TrainConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg: DictConfig = hydra.compose(config_name="train", overrides=overrides)
    data = OmegaConf.to_container(cfg, resolve=True)
    return TrainConfig(
        **{k: Path(v) if k.endswith("_dir") or k in {"data_dir", "resume", "checkpoint_dir"} and v is not None else v for k, v in data.items() if k != "tracking"},
        tracking=data["tracking"],
    )


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        cfg = _build_train_config(args.overrides)
        if isinstance(cfg.tracking, dict):
            from src.config import TrackingConfig

            cfg.tracking = TrackingConfig(**cfg.tracking)
        train(cfg)
        return

    if args.command == "eval":
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        evaluate(
            checkpoint=Path(args.checkpoint),
            real_dir=Path(args.real_dir),
            output_dir=Path(args.output_dir),
            sample_count=args.sample_count,
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            seed=args.seed,
            num_workers=args.num_workers,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
            device=device,
            epoch=args.epoch,
        )


if __name__ == "__main__":
    main()
