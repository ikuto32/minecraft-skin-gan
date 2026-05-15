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


def _parse_seed_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return [int(x) for x in items]
    if isinstance(value, list):
        return [int(x) for x in value]
    raise ValueError(f"Unsupported seed list type: {type(value)!r}")


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    tracking = TrackingConfig(**data["tracking"])
    data.pop("mode", None)
    data.pop("tracking", None)
    data["eval_seeds"] = _parse_seed_list(data.get("eval_seeds"))
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
    data["seeds"] = _parse_seed_list(data.get("seeds"))
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
