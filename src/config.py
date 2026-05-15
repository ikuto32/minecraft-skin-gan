from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


class ConfigValidationError(ValueError):
    def __init__(self, title: str, details: list[dict[str, object]]) -> None:
        super().__init__(title)
        self.title = title
        self.details = details

    def __str__(self) -> str:
        detail_str = "; ".join(
            f"{d['path']}: {d['message']}" for d in self.details
        )
        return f"{self.title}: {detail_str}" if detail_str else self.title


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrackingConfig(SchemaModel):
    backend: Literal["none", "wandb", "mlflow"] = "none"
    project: str = "minecraft-skin-gan"
    run_name: str | None = None
    entity: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "minecraft-skin-gan"


class TrainConfig(SchemaModel):
    model_name: str = "dcgan_baseline"
    data_dir: Path = Path("data/skins")
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=64, gt=0)
    z_dim: int = Field(default=100, gt=0)
    lr: float = 2e-4
    resume: Path | None = None
    auto_resume: bool = True
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    r1_gamma: float = 10.0
    r1_interval: int = Field(default=16, gt=0)
    compile: bool = False
    channels_last: bool = False
    amp_dtype: Literal["none", "float16", "bfloat16"] = "bfloat16"
    performance_profile: Literal["auto", "safe", "max"] = "auto"
    num_workers: int = Field(default=2, ge=0)
    persistent_workers: bool = False
    prefetch_factor: int | None = Field(default=2, gt=0)
    eval_every: int = Field(default=10, gt=0)
    eval_sample_count: int = Field(default=2048, gt=0)
    eval_seed: int = Field(default=1234, gt=0)
    eval_seeds: list[int] | None = None
    eval_batch_size: int = Field(default=64, gt=0)
    eval_num_workers: int = Field(default=2, ge=0)
    eval_output_dir: Path = Path("outputs/eval")
    kid_subsets: int = Field(default=50, gt=0)
    kid_subset_size: int = Field(default=32, gt=0)
    ema_beta: float = 0.999
    use_ada: bool = False
    ada_target: float = 0.6
    ada_interval: int = Field(default=4, gt=0)
    ada_speed: float = 0.001
    checkpoint_dir: Path = Path("checkpoints")
    best_metric: Literal["fid", "kid_mean"] = "fid"
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> "TrainConfig":
        _validate_positive("epochs", self.epochs)
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("z_dim", self.z_dim)
        _validate_positive("r1_interval", self.r1_interval)
        _validate_positive("eval_every", self.eval_every)
        _validate_positive("eval_sample_count", self.eval_sample_count)
        _validate_positive("eval_seed", self.eval_seed)
        _validate_positive("eval_batch_size", self.eval_batch_size)
        _validate_positive("kid_subsets", self.kid_subsets)
        _validate_positive("kid_subset_size", self.kid_subset_size)
        _validate_positive("ada_interval", self.ada_interval)
        _validate_non_negative("num_workers", self.num_workers)
        _validate_non_negative("eval_num_workers", self.eval_num_workers)
        if self.num_workers == 0:
            if self.persistent_workers:
                raise ValueError(
                    "persistent_workers must be false when num_workers is 0"
                )
            if self.prefetch_factor is not None:
                raise ValueError("prefetch_factor must be null when num_workers is 0")
        else:
            if self.prefetch_factor is None:
                raise ValueError("prefetch_factor is required when num_workers > 0")

        if self.performance_profile == "safe" and self.amp_dtype == "float16":
            raise ValueError(
                "amp_dtype=float16 is not allowed with performance_profile=safe"
            )
        return self


class EvalConfig(SchemaModel):
    model_name: str = "dcgan_baseline"
    checkpoint: Path = Path("checkpoints/latest.pt")
    real_dir: Path = Path("data/skins")
    output_dir: Path = Path("outputs/eval")
    sample_count: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=64, gt=0)
    z_dim: int = Field(default=100, gt=0)
    seed: int = Field(default=1234, gt=0)
    deterministic: bool = True
    cudnn_benchmark: bool = False
    seeds: list[int] | None = None
    num_workers: int = Field(default=2, ge=0)
    kid_subsets: int = Field(default=50, gt=0)
    kid_subset_size: int = Field(default=32, gt=0)
    device: Literal["cpu", "cuda"] | None = None
    epoch: int | None = None
    resize: int = Field(default=64, gt=0)
    color_mode: Literal["RGB", "RGBA"] = "RGBA"
    reuse_real_features: bool = True


    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> "EvalConfig":
        _validate_positive("sample_count", self.sample_count)
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("z_dim", self.z_dim)
        _validate_positive("seed", self.seed)
        _validate_positive("kid_subsets", self.kid_subsets)
        _validate_positive("kid_subset_size", self.kid_subset_size)
        _validate_positive("resize", self.resize)
        _validate_non_negative("num_workers", self.num_workers)
        return self


def format_validation_error(exc: ValidationError, *, root: str) -> ConfigValidationError:
    details: list[dict[str, object]] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        details.append({"path": f"{root}.{loc}" if loc else root, "message": err["msg"]})
    return ConfigValidationError(f"Invalid {root} config", details)
