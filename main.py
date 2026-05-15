from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from src.config import EvalConfig, TrainConfig, format_validation_error
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


def _mode_payload(cfg: DictConfig) -> DictConfig:
    mode_cfg = cfg.get("mode")
    if isinstance(mode_cfg, DictConfig):
        return mode_cfg
    return cfg


def _mode_name(cfg: DictConfig) -> str | None:
    mode_cfg = cfg.get("mode")
    if isinstance(mode_cfg, DictConfig):
        mode_value = mode_cfg.get("mode")
        return str(mode_value) if mode_value is not None else None
    if isinstance(mode_cfg, str):
        return mode_cfg
    return None


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(_mode_payload(cfg), resolve=True)
    assert isinstance(data, dict)
    data.pop("mode", None)
    data["eval_seeds"] = _parse_seed_list(data.get("eval_seeds"))
    try:
        return TrainConfig.model_validate(data)
    except ValidationError as exc:
        formatted = format_validation_error(exc, root="train")
        raise SystemExit(json.dumps({"error": formatted.title, "details": formatted.details}, ensure_ascii=False)) from exc


def _build_eval_config(cfg: DictConfig) -> EvalConfig:
    data = OmegaConf.to_container(_mode_payload(cfg), resolve=True)
    assert isinstance(data, dict)
    data.pop("mode", None)
    data["seeds"] = _parse_seed_list(data.get("seeds"))
    try:
        return EvalConfig.model_validate(data)
    except ValidationError as exc:
        formatted = format_validation_error(exc, root="eval")
        raise SystemExit(json.dumps({"error": formatted.title, "details": formatted.details}, ensure_ascii=False)) from exc


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mode_name = _mode_name(cfg)
    if mode_name == "train":
        train_cfg = _build_train_config(cfg)
        _validate_composed_train_config(train_cfg)
        train(train_cfg)
        return
    if mode_name == "eval":
        evaluate(_build_eval_config(cfg))
        return
    raise ValueError(f"Unsupported mode: {mode_name!r}")


if __name__ == "__main__":
    main()
