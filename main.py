from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import EvalConfig, TrackingConfig, TrainConfig
from src.engine import train
from src.eval import evaluate


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    tracking = TrackingConfig(**data["tracking"])
    data.pop("mode", None)
    data.pop("tracking", None)
    return TrainConfig(**data, tracking=tracking)


def _build_eval_config(cfg: DictConfig) -> EvalConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    data.pop("mode", None)
    return EvalConfig(**data)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "train":
        train(_build_train_config(cfg))
        return
    if cfg.mode == "eval":
        evaluate(_build_eval_config(cfg))
        return
    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
