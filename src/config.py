from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackingConfig:
    backend: str = "none"  # none|wandb|mlflow
    project: str = "minecraft-skin-gan"
    run_name: str | None = None
    entity: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "minecraft-skin-gan"


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/skins")
    epochs: int = 100
    batch_size: int = 64
    z_dim: int = 100
    lr: float = 2e-4
    resume: Path | None = None
    auto_resume: bool = True
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
    ema_beta: float = 0.999
    use_ada: bool = False
    ada_target: float = 0.6
    ada_interval: int = 4
    ada_speed: float = 0.001
    checkpoint_dir: Path = Path("checkpoints")
    best_metric: str = "fid"  # fid|kid_mean
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
