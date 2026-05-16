from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import importlib.metadata
import platform

import torch

from src.config import TrackingConfig


class Tracker:
    def __init__(
        self,
        cfg: TrackingConfig,
        full_config: dict[str, Any],
        *,
        runtime_profile: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = cfg.backend.lower()
        self.run = None

        common_tags = self._build_common_tags(full_config, runtime_profile=runtime_profile)

        if self.backend == "wandb":
            import wandb

            wandb_config = dict(full_config)
            wandb_config["tracking_context"] = common_tags
            self.run = wandb.init(
                project=cfg.project,
                name=cfg.run_name,
                entity=cfg.entity,
                config=wandb_config,
                tags=self._build_wandb_tags(common_tags),
            )
            for key, value in common_tags.items():
                wandb.run.summary[key] = value
            wandb.define_metric("train/step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="epoch")
            wandb.define_metric("eval/precision", step_metric="epoch")
            wandb.define_metric("eval/recall", step_metric="epoch")
        elif self.backend == "mlflow":
            import mlflow

            if cfg.mlflow_tracking_uri:
                mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            self.run = mlflow.start_run(run_name=cfg.run_name)
            mlflow.log_params(_flatten(full_config))
            mlflow.log_params(common_tags)
            mlflow.set_tags(common_tags)

    def log_metrics(self, metrics: dict[str, float | int], *, step: int | None = None, epoch: int | None = None) -> None:
        payload = dict(metrics)
        if step is not None:
            payload["train/step"] = step
        if epoch is not None:
            payload["epoch"] = epoch

        if self.backend == "wandb":
            import wandb

            wandb.log(payload, step=step)
        elif self.backend == "mlflow":
            import mlflow

            numeric_payload = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric_payload, step=step if step is not None else epoch)

    def log_image(self, key: str, image_path: Path, step: int) -> None:
        if self.backend == "wandb":
            import wandb

            wandb.log({key: wandb.Image(str(image_path))}, step=step)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_artifact(str(image_path), artifact_path=f"images/step_{step}")

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.backend == "wandb":
            import wandb

            artifact = wandb.Artifact(name=f"run-artifacts-{wandb.run.id}", type="run-data")
            artifact.add_file(str(path), name=f"{artifact_path}/{path.name}" if artifact_path else path.name)
            wandb.log_artifact(artifact)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_histogram(self, key: str, values: torch.Tensor, step: int) -> None:
        if self.backend == "wandb":
            import wandb

            wandb.log({key: wandb.Histogram(values.detach().flatten().cpu().numpy()), "train/step": step}, step=step)

    def log_summary(self, values: dict[str, float | int]) -> None:
        if self.backend == "wandb":
            import wandb

            for key, value in values.items():
                wandb.run.summary[key] = value
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_metrics({f"summary/{k}": float(v) for k, v in values.items()})

    def close(self) -> None:
        if self.backend == "wandb":
            import wandb

            wandb.finish()
        elif self.backend == "mlflow":
            import mlflow

            mlflow.end_run()

    def _build_common_tags(
        self,
        full_config: dict[str, Any],
        *,
        runtime_profile: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        train_cfg = full_config.get("train", full_config)
        data_dir = train_cfg.get("data_dir")
        eval_seeds = train_cfg.get("eval_seeds")
        dataset_size = self._count_dataset_size(data_dir)
        runtime_profile = runtime_profile or {}
        metadata = {
            "env/python_version": platform.python_version(),
            "env/torch_version": torch.__version__,
            "env/cuda_version": str(torch.version.cuda or "none"),
            "env/device_name": self._get_device_name(),
            "env/torchvision_version": self._pkg_version("torchvision"),
            "env/numpy_version": self._pkg_version("numpy"),
            "env/pillow_version": self._pkg_version("Pillow"),
            "data/data_dir": str(data_dir),
            "data/dataset_size": str(train_cfg.get("dataset_size", dataset_size)),
            "data/dataset_fingerprint": self._dataset_fingerprint(data_dir),
            "runtime/profile": str(train_cfg.get("performance_profile", "unknown")),
            "runtime/compile": str(runtime_profile.get("compile", train_cfg.get("compile"))),
            "runtime/amp_dtype": str(runtime_profile.get("amp_dtype", train_cfg.get("amp_dtype"))),
            "runtime/channels_last": str(runtime_profile.get("channels_last", "unknown")),
            "runtime/eval_seeds": "none" if not eval_seeds else ",".join(str(v) for v in eval_seeds),
            "runtime/seed": str(train_cfg.get("seed")),
            "runtime/model_type": str(train_cfg.get("model_type", "dcgan")),
        }
        return metadata


    @staticmethod
    def _build_wandb_tags(common_tags: dict[str, str]) -> list[str]:
        tags: list[str] = []
        for key, value in common_tags.items():
            tag = f"{key}:{value}"
            if len(tag) <= 64:
                tags.append(tag)
                continue

            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
            shortened = f"{key}:sha256:{digest}"
            tags.append(shortened if len(shortened) <= 64 else f"sha256:{digest}")
        return tags

    @staticmethod
    def _count_dataset_size(data_dir: Any) -> int:
        if data_dir is None:
            return -1
        path = Path(data_dir)
        if not path.exists() or not path.is_dir():
            return -1
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)

    @staticmethod
    def _pkg_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    @staticmethod
    def _get_device_name() -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
        return "cpu"

    @staticmethod
    def _dataset_fingerprint(data_dir: Any) -> str:
        if data_dir is None:
            return "none"
        path = Path(data_dir)
        if not path.exists() or not path.is_dir():
            return "invalid"

        manifest: list[str] = []
        for p in sorted(path.rglob("*")):
            if not p.is_file():
                continue
            stat = p.stat()
            rel = p.relative_to(path).as_posix()
            manifest.append(f"{rel}\t{stat.st_size}\t{int(stat.st_mtime_ns)}")

        digest = hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
