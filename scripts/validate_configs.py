from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError

from src.config import EvalConfig, TrainConfig, format_validation_error


def _validate_mode(mode: str) -> list[dict[str, object]]:
    cfg = compose(config_name="config", overrides=[f"mode={mode}"])
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    data.pop("mode", None)

    try:
        if mode == "train":
            TrainConfig.model_validate(data)
        else:
            EvalConfig.model_validate(data)
        return []
    except ValidationError as exc:
        err = format_validation_error(exc, root=mode)
        return [{"mode": mode, "error": err.title, "details": err.details}]


def main() -> int:
    errors: list[dict[str, object]] = []
    with initialize_config_dir(config_dir=str(Path("conf").resolve()), version_base=None):
        errors.extend(_validate_mode("train"))
        errors.extend(_validate_mode("eval"))

    if errors:
        print(json.dumps({"status": "invalid", "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"status": "ok", "validated_modes": ["train", "eval"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
