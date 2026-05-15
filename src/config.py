from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/skins")
    epochs: int = 100
    batch_size: int = 64
    z_dim: int = 100
    lr: float = 2e-4
    resume: Path | None = None
    seed: int = 42
    r1_gamma: float = 10.0
    r1_interval: int = 16
    compile: bool = False
    channels_last: bool = False
    amp_dtype: str = "bfloat16"
    num_workers: int = 2
    persistent_workers: bool = False
    prefetch_factor: int = 2
    eval_every: int = 10
    eval_sample_count: int = 2048
    eval_seed: int = 1234
    eval_batch_size: int = 64
    eval_num_workers: int = 2
    eval_output_dir: Path = Path("outputs/eval")
    kid_subsets: int = 50
    kid_subset_size: int = 32


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/skins"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    parser.add_argument("--r1-interval", type=int, default=16)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--amp-dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-sample-count", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--eval-output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=32)
    return TrainConfig(**vars(parser.parse_args()))
