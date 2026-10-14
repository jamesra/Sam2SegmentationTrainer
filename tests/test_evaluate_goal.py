from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch import nn

from sam2_segmentation_trainer.evaluate import EvalRequest, evaluate

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


def _val_logits(model, batch, device, image_size):
    """The ground truth of each variant's image, as logits: a perfect model."""
    return batch["mask"][batch["variant_index"]].float() * 20.0 - 10.0


class EvaluateGoalTests(unittest.TestCase):
    def test_goal_run_scores_every_family_on_the_full_split_and_writes_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_crops(root, "RC1", n=12)
            request = EvalRequest(
                finetuned=None, base_ckpt=root / "base.pt", model_cfg="unused", data_dir=root, volumes=["RC1"],
                split_path=root / "runs" / "v4" / "split.json", image_size=32, threshold=0.5, out_dir=root / "out",
                grid_every=0, max_grids=0, num_workers=0, goal=True,
                overrides=("data.split_mode=location", "data.train_split=0.5", "val.core_band_px=2", "data.require_circle_file=false"),
            )
            out = io.StringIO()
            with mock.patch("sam2_segmentation_trainer.evaluate._load_model", return_value=nn.Linear(1, 1)), mock.patch(
                "sam2_segmentation_trainer.validation.predict_variants", _val_logits
            ), contextlib.redirect_stdout(out):
                summary = evaluate(request)
            self.assertTrue((root / "runs" / "v4" / "split.json").is_file())  # created for the run to reuse
            written = json.loads((root / "out" / "goal_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["model"], "zero-shot")
        self.assertEqual(
            set(written["families"]), {"circle_mask", "correction", "near_correct", "point", "centroid"}
        )
        self.assertGreater(written["samples"], 0)
        for family in written["families"].values():
            self.assertIn("core_recall", family["goal"])
        self.assertAlmostEqual(written["families"]["circle_mask"]["goal"]["core_recall"], 1.0)
        self.assertIn("val [circle_mask", out.getvalue())


if __name__ == "__main__":
    unittest.main()
