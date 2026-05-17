import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.config import EvalConfig
from src.eval_metrics import compute_metrics


class ConstantGenerator(torch.nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        b = noise.size(0)
        return torch.zeros((b, 3, 64, 64), device=noise.device)


class EvalMetricsPRSamplingTest(unittest.TestCase):
    def test_pr_uses_sample_count_for_real_and_fake_sets(self) -> None:
        dataset = torch.rand(8, 3, 64, 64)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        with TemporaryDirectory() as tmp_dir:
            cfg = EvalConfig(
                checkpoint=Path(tmp_dir) / "checkpoint.pt",
                real_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir),
                sample_count=5,
                enable_fid=False,
                enable_kid=False,
                enable_precision_recall=True,
                z_dim=8,
            )
            generator = ConstantGenerator(z_dim=8)

            def _fake_pr_metric(**kwargs):
                real_dir = Path(kwargs["input1"])
                fake_dir = Path(kwargs["input2"])
                real_files = sorted(real_dir.glob("*.png"))
                fake_files = sorted(fake_dir.glob("*.png"))
                self.assertEqual(len(real_files), cfg.sample_count)
                self.assertEqual(len(fake_files), cfg.sample_count)

                with Image.open(real_files[0]) as sample_real:
                    self.assertEqual(sample_real.size, (299, 299))
                    self.assertEqual(sample_real.mode, "RGB")
                with Image.open(fake_files[0]) as sample_fake:
                    self.assertEqual(sample_fake.size, (299, 299))
                    self.assertEqual(sample_fake.mode, "RGB")
                return {"precision": 0.3, "recall": 0.4}

            with patch("src.eval_metrics.calculate_metrics", side_effect=_fake_pr_metric):
                result = compute_metrics(
                    cfg=cfg,
                    loader=loader,
                    generator=generator,
                    device="cpu",
                )

        self.assertEqual(result.precision, 0.3)
        self.assertEqual(result.recall, 0.4)


if __name__ == "__main__":
    unittest.main()
