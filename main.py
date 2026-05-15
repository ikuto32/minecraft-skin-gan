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


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    data.pop("mode", None)
    data["eval_seeds"] = _parse_seed_list(data.get("eval_seeds"))
    try:
        return TrainConfig.model_validate(data)
    except ValidationError as exc:
        formatted = format_validation_error(exc, root="train")
        raise SystemExit(json.dumps({"error": formatted.title, "details": formatted.details}, ensure_ascii=False)) from exc


def _build_eval_config(cfg: DictConfig) -> EvalConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
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
    if cfg.mode == "train":
        train(train_cfg := _build_train_config(cfg))
        return
    if cfg.mode == "eval":
        evaluate(_build_eval_config(cfg))
        return
    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
