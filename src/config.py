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
    backend: Literal["none", "wandb"] = "none"
    project: str = "minecraft-skin-gan"
    run_name: str | None = None
    entity: str | None = None


class LossConfig(SchemaModel):
    name: Literal["hinge", "wgan_gp", "logistic_r1", "r3gan_relativistic"] = "hinge"
    gp_lambda: float = 10.0
    r1_gamma: float = 10.0
    r1_interval: int = Field(default=16, gt=0)
    r2_gamma: float = 0.0
    r2_interval: int = Field(default=16, gt=0)
    rel_scale: float = 1.0
    rel_margin: float = 0.0

    @model_validator(mode="after")
    def _validate_loss(self) -> "LossConfig":
        if self.name == "wgan_gp" and self.gp_lambda < 0:
            raise ValueError("loss.gp_lambda must be >= 0 for wgan_gp")
        if self.name == "logistic_r1" and self.r1_interval <= 0:
            raise ValueError("loss.r1_interval must be > 0 for logistic_r1")
        if self.name == "logistic_r1" and self.r2_interval <= 0:
            raise ValueError("loss.r2_interval must be > 0 for logistic_r1")
        if self.name == "r3gan_relativistic" and self.rel_scale <= 0:
            raise ValueError("loss.rel_scale must be > 0 for r3gan_relativistic")
        return self


class TrainConfig(SchemaModel):
    model_name: str = "dcgan_baseline"
    data_dir: Path = Path("data/skins")
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=64, gt=0)
    n_critic: int = Field(default=1, ge=1)
    z_dim: int = Field(default=100, gt=0)
    lr: float = 2e-4
    lr_g: float | None = None
    lr_d: float | None = None
    resume: Path | None = None
    auto_resume: bool = True
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
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
    eval_enable_fid: bool = True
    eval_enable_kid: bool = True
    eval_enable_precision_recall: bool = True
    ema_beta: float = 0.999
    use_ada: bool = False
    ada_target: float = 0.6
    ada_grad_target: float = 0.2
    ada_sign_weight: float = 0.7
    ada_grad_weight: float = 0.3
    ada_interval: int = Field(default=4, gt=0)
    ada_speed: float = 0.001
    ada_policy: str = "flip,noise,color,translation,cutout"
    checkpoint_dir: Path = Path("checkpoints")
    best_metric: Literal["fid", "kid_mean", "pr_tradeoff"] = "fid"
    pr_recall_floor: float = 0.6
    loss: LossConfig = Field(default_factory=LossConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> "TrainConfig":
        _validate_positive("epochs", self.epochs)
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("z_dim", self.z_dim)
        _validate_positive("loss.r1_interval", self.loss.r1_interval)
        _validate_positive("eval_every", self.eval_every)
        _validate_positive("eval_sample_count", self.eval_sample_count)
        _validate_positive("eval_seed", self.eval_seed)
        _validate_positive("eval_batch_size", self.eval_batch_size)
        _validate_positive("kid_subsets", self.kid_subsets)
        _validate_positive("kid_subset_size", self.kid_subset_size)
        if not (self.eval_enable_fid or self.eval_enable_kid or self.eval_enable_precision_recall):
            raise ValueError("At least one eval metric must be enabled for periodic eval")
        _validate_positive("ada_interval", self.ada_interval)
        if not 0.0 <= self.ada_target <= 1.0:
            raise ValueError("ada_target must be within [0, 1]")
        if self.ada_grad_target <= 0:
            raise ValueError("ada_grad_target must be > 0")
        if self.ada_sign_weight < 0 or self.ada_grad_weight < 0:
            raise ValueError("ada_sign_weight and ada_grad_weight must be >= 0")
        if (self.ada_sign_weight + self.ada_grad_weight) <= 0:
            raise ValueError("sum of ada_sign_weight and ada_grad_weight must be > 0")
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
        if self.lr_g is None:
            self.lr_g = self.lr
        if self.lr_d is None:
            self.lr_d = self.lr
        if self.lr_g <= 0 or self.lr_d <= 0:
            raise ValueError("lr_g and lr_d must be > 0")
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
    enable_fid: bool = True
    enable_kid: bool = True
    enable_precision_recall: bool = True
    device: Literal["cpu", "cuda"] | None = None
    epoch: int | None = None
    resize: int = Field(default=64, gt=0)
    color_mode: Literal["RGB", "RGBA"] = "RGBA"

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> "EvalConfig":
        _validate_positive("sample_count", self.sample_count)
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("z_dim", self.z_dim)
        _validate_positive("seed", self.seed)
        _validate_positive("kid_subsets", self.kid_subsets)
        _validate_positive("kid_subset_size", self.kid_subset_size)
        _validate_positive("resize", self.resize)
        if not (self.enable_fid or self.enable_kid or self.enable_precision_recall):
            raise ValueError("At least one eval metric must be enabled")
        _validate_non_negative("num_workers", self.num_workers)
        return self


def format_validation_error(exc: ValidationError, *, root: str) -> ConfigValidationError:
    details: list[dict[str, object]] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        details.append({"path": f"{root}.{loc}" if loc else root, "message": err["msg"]})
    return ConfigValidationError(f"Invalid {root} config", details)
