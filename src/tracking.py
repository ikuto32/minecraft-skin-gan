from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import TrackingConfig


class Tracker:
    def __init__(self, cfg: TrackingConfig, full_config: dict[str, Any]) -> None:
        self.cfg = cfg
        self.backend = cfg.backend.lower()
        self.run = None

        common_tags = self._build_common_tags(full_config)

        if self.backend == "wandb":
            import wandb

            self.run = wandb.init(
                project=cfg.project,
                name=cfg.run_name,
                entity=cfg.entity,
                config=full_config,
                tags=[f"{k}:{v}" for k, v in common_tags.items()],
            )
            wandb.define_metric("train/step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="epoch")
        elif self.backend == "mlflow":
            import mlflow

            if cfg.mlflow_tracking_uri:
                mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            self.run = mlflow.start_run(run_name=cfg.run_name)
            mlflow.log_params(_flatten(full_config))
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

    def _build_common_tags(self, full_config: dict[str, Any]) -> dict[str, str]:
        train_cfg = full_config.get("train", full_config)
        data_dir = train_cfg.get("data_dir")
        eval_seeds = train_cfg.get("eval_seeds")
        return {
            "model_type": str(train_cfg.get("model_type", "dcgan")),
            "dataset_size": str(train_cfg.get("dataset_size", self._count_dataset_size(data_dir))),
            "seed": str(train_cfg.get("seed")),
            "amp_dtype": str(train_cfg.get("amp_dtype")),
            "compile": str(train_cfg.get("compile")),
            "eval_seeds": "none" if not eval_seeds else ",".join(str(v) for v in eval_seeds),
        }

    @staticmethod
    def _count_dataset_size(data_dir: Any) -> int:
        if data_dir is None:
            return -1
        path = Path(data_dir)
        if not path.exists() or not path.is_dir():
            return -1
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
