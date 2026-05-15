from __future__ import annotations

import argparse

from src.config import TrainConfig
from src.engine import train
from src.eval import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minecraft skin GAN training/evaluation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DCGAN model")
    train_parser.add_argument("--data-dir", default="data/skins")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--z-dim", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--r1-gamma", type=float, default=10.0)
    train_parser.add_argument("--r1-interval", type=int, default=16)
    train_parser.add_argument("--compile", action="store_true")
    train_parser.add_argument("--channels-last", action="store_true")
    train_parser.add_argument("--amp-dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--persistent-workers", action="store_true")
    train_parser.add_argument("--prefetch-factor", type=int, default=2)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--eval-sample-count", type=int, default=2048)
    train_parser.add_argument("--eval-seed", type=int, default=1234)
    train_parser.add_argument("--eval-batch-size", type=int, default=64)
    train_parser.add_argument("--eval-num-workers", type=int, default=2)
    train_parser.add_argument("--eval-output-dir", default="outputs/eval")
    train_parser.add_argument("--kid-subsets", type=int, default=50)
    train_parser.add_argument("--kid-subset-size", type=int, default=32)

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = TrainConfig(**vars(args))
        train(config)
        return

    if args.command == "eval":
        import torch
        from pathlib import Path

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
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
