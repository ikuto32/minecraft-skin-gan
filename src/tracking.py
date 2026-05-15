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

        if self.backend == "wandb":
            import wandb

            self.run = wandb.init(project=cfg.project, name=cfg.run_name, entity=cfg.entity, config=full_config)
            wandb.define_metric('train/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('eval/*', step_metric='train/step')
        elif self.backend == "mlflow":
            import mlflow

            if cfg.mlflow_tracking_uri:
                mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)
            self.run = mlflow.start_run(run_name=cfg.run_name)
            mlflow.log_params(_flatten(full_config))

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None:
        if self.backend == "wandb":
            import wandb

            wandb.log(metrics | {"train/step": step}, step=step)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_image(self, key: str, image_path: Path, step: int) -> None:
        if self.backend == "wandb":
            import wandb

            wandb.log({key: wandb.Image(str(image_path))}, step=step)
        elif self.backend == "mlflow":
            import mlflow

            mlflow.log_artifact(str(image_path), artifact_path=f"images/step_{step}")


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


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
