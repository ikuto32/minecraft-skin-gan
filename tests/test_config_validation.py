import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from scripts import validate_configs
from src.config import TrainConfig


class TrainConfigValidationTest(unittest.TestCase):
    def test_num_workers_zero_allows_null_prefetch_factor(self) -> None:
        cfg = TrainConfig(num_workers=0, prefetch_factor=None)
        self.assertEqual(cfg.num_workers, 0)
        self.assertIsNone(cfg.prefetch_factor)

    def test_non_wgan_gp_accepts_gp_lambda(self) -> None:
        cfg = TrainConfig(loss={"name": "hinge", "gp_lambda": -123.0})
        self.assertEqual(cfg.loss.name, "hinge")
        self.assertEqual(cfg.loss.gp_lambda, -123.0)

    def test_wgan_gp_rejects_negative_gp_lambda(self) -> None:
        with self.assertRaises(ValidationError):
            TrainConfig(loss={"name": "wgan_gp", "gp_lambda": -0.1})

    def test_num_workers_zero_rejects_prefetch_factor(self) -> None:
        with self.assertRaises(ValidationError):
            TrainConfig(num_workers=0, prefetch_factor=2)

    def test_eval_seeds_rejects_empty_string(self) -> None:
        with self.assertRaises(ValidationError):
            TrainConfig.model_validate({"eval_seeds": ""})

    def test_eval_seeds_rejects_non_numeric_values(self) -> None:
        with self.assertRaises(ValidationError):
            TrainConfig.model_validate({"eval_seeds": ["1", "oops"]})

    def test_sample_every_kimg_rejects_negative(self) -> None:
        with self.assertRaises(ValidationError):
            TrainConfig.model_validate({"sample_every_kimg": -0.5})


class ValidateConfigsCliEntrypointTest(unittest.TestCase):
    def test_main_returns_zero_for_repository_configs(self) -> None:
        for config_path in (
            Path("conf/config.yaml"),
            Path("conf/mode/train.yaml"),
            Path("conf/mode/eval.yaml"),
        ):
            self.assertTrue(config_path.exists(), f"missing fixture config: {config_path}")

        rc = validate_configs.main()
        self.assertEqual(rc, 0)

    def test_main_returns_one_when_any_mode_is_invalid(self) -> None:
        with patch.object(validate_configs, "_validate_mode") as validate_mode:
            validate_mode.side_effect = [
                [{"mode": "train", "error": "Invalid train config", "details": []}],
                [],
            ]
            rc = validate_configs.main()

        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
