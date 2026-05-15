from dataclasses import dataclass, field
from pathlib import Path


def _ensure_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


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

    def __post_init__(self) -> None:
        self.data_dir = _ensure_path(self.data_dir)  # type: ignore[assignment]
        self.resume = _ensure_path(self.resume)  # type: ignore[assignment]
        self.eval_output_dir = _ensure_path(self.eval_output_dir)  # type: ignore[assignment]
        self.checkpoint_dir = _ensure_path(self.checkpoint_dir)  # type: ignore[assignment]

        _validate_positive("epochs", self.epochs)
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("z_dim", self.z_dim)
        _validate_positive("num_workers", self.num_workers)


@dataclass
class EvalConfig:
    checkpoint: Path = Path("checkpoints/latest.pt")
    real_dir: Path = Path("data/skins")
    output_dir: Path = Path("outputs/eval")
    sample_count: int = 2048
    batch_size: int = 64
    z_dim: int = 100
    seed: int = 1234
    num_workers: int = 2
    kid_subsets: int = 50
    kid_subset_size: int = 32
    device: str | None = None
    epoch: int | None = None

    def __post_init__(self) -> None:
        self.checkpoint = _ensure_path(self.checkpoint)  # type: ignore[assignment]
        self.real_dir = _ensure_path(self.real_dir)  # type: ignore[assignment]
        self.output_dir = _ensure_path(self.output_dir)  # type: ignore[assignment]

        _validate_positive("sample_count", self.sample_count)
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("z_dim", self.z_dim)
        _validate_positive("num_workers", self.num_workers)
        _validate_positive("kid_subsets", self.kid_subsets)
        _validate_positive("kid_subset_size", self.kid_subset_size)

        if self.device not in {None, "cpu", "cuda"}:
            raise ValueError("device must be one of: null, 'cpu', 'cuda'")
