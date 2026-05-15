from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _ensure_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass
class TrackingConfig:
    backend: Literal["none", "wandb", "mlflow"] = "none"
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
    amp_dtype: Literal["none", "float16", "bfloat16"] = "bfloat16"
    performance_profile: Literal["auto", "safe", "max"] = "auto"
    num_workers: int = 2
    persistent_workers: bool = False
    prefetch_factor: int = 2
    eval_every: int = 10
    eval_sample_count: int = 2048
    eval_seed: int = 1234
    eval_seeds: list[int] | None = None
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
    best_metric: Literal["fid", "kid_mean"] = "fid"
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
        _validate_positive("r1_interval", self.r1_interval)
        _validate_positive("eval_every", self.eval_every)
        _validate_positive("eval_sample_count", self.eval_sample_count)
        _validate_positive("eval_batch_size", self.eval_batch_size)
        _validate_positive("eval_num_workers", self.eval_num_workers)
        _validate_positive("kid_subsets", self.kid_subsets)
        _validate_positive("kid_subset_size", self.kid_subset_size)
        _validate_positive("ada_interval", self.ada_interval)


def validate_tracking_config(cfg: TrackingConfig) -> None:
    if cfg.backend not in {"none", "wandb", "mlflow"}:
        raise ValueError(
            "tracking.backend must be one of: 'none', 'wandb', 'mlflow'"
        )


def validate_train_config(cfg: TrainConfig) -> None:
    validate_tracking_config(cfg.tracking)
    if cfg.best_metric not in {"fid", "kid_mean"}:
        raise ValueError("best_metric must be one of: 'fid', 'kid_mean'")
    if cfg.amp_dtype not in {"none", "float16", "bfloat16"}:
        raise ValueError("amp_dtype must be one of: 'none', 'float16', 'bfloat16'")
    if cfg.performance_profile not in {"auto", "safe", "max"}:
        raise ValueError("performance_profile must be one of: 'auto', 'safe', 'max'")


@dataclass
class EvalConfig:
    checkpoint: Path = Path("checkpoints/latest.pt")
    real_dir: Path = Path("data/skins")
    output_dir: Path = Path("outputs/eval")
    sample_count: int = 2048
    batch_size: int = 64
    z_dim: int = 100
    seed: int = 1234
    seeds: list[int] | None = None
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
        _validate_positive("seed", self.seed)
        if self.seeds is not None:
            if not self.seeds:
                raise ValueError("seeds must not be empty")
            for seed in self.seeds:
                _validate_positive("seeds[*]", seed)

        if self.device not in {None, "cpu", "cuda"}:
            raise ValueError("device must be one of: null, 'cpu', 'cuda'")
