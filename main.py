from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import (
    EvalConfig,
    TrackingConfig,
    TrainConfig,
    validate_tracking_config,
    validate_train_config,
)
from src.engine import train
from src.eval import evaluate


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    tracking = TrackingConfig(**data["tracking"])
    data.pop("mode", None)
    data.pop("tracking", None)
    return TrainConfig(**data, tracking=tracking)


def _validate_composed_train_config(cfg: TrainConfig) -> None:
    try:
        validate_tracking_config(cfg.tracking)
        validate_train_config(cfg)
    except ValueError as exc:
        raise SystemExit(f"Invalid train config: {exc}") from exc


def _build_eval_config(cfg: DictConfig) -> EvalConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    data.pop("mode", None)
    return EvalConfig(**data)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "train":
        train_cfg = _build_train_config(cfg)
        _validate_composed_train_config(train_cfg)
        train(train_cfg)
        return
    if cfg.mode == "eval":
        evaluate(_build_eval_config(cfg))
        return
    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
