from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import TrackingConfig


class Tracker:
    def __init__(self, cfg: TrackingConfig, full_config: dict[str, Any]) -> None:
        self.cfg = cfg
        self.backend = cfg.backend.lower()
        self.run = None

        if self.backend == "wandb":
            import wandb

            self.run = wandb.init(project=cfg.project, name=cfg.run_name, entity=cfg.entity, config=full_config)
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

            wandb.log(metrics, step=step)
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
