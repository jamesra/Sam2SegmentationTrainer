from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch import nn

from annotation_crops import prompt_encoding as enc
from sam2_segmentation_trainer.export_serve import export_serve_checkpoint, finetuned_state_dict, serve_metadata


class ExportServeTests(unittest.TestCase):
    def test_raw_state_dict(self) -> None:
        raw = {"image_encoder.trunk.weight": 1, "sam_mask_decoder.bias": 2}
        self.assertIs(finetuned_state_dict(raw), raw)

    def test_meta_wrapper(self) -> None:
        inner = {"image_encoder.trunk.weight": 1}
        self.assertEqual(finetuned_state_dict({"model": inner}), inner)

    def test_rejects_non_dict(self) -> None:
        with self.assertRaises(TypeError):
            finetuned_state_dict([1, 2, 3])

    def test_rejects_unrecognized_dict(self) -> None:
        with self.assertRaises(ValueError):
            finetuned_state_dict({"optimizer": {}})

    def test_metadata_holds_every_encoding_constant(self) -> None:
        meta = serve_metadata(0.9)
        self.assertEqual(
            meta,
            {
                "prompt_encoding": enc.PROMPT_ENCODING_VERSION,
                "EDGE_TAU_PX": enc.EDGE_TAU_PX,
                "EDGE_TAU_DEPTH_FRAC": enc.EDGE_TAU_DEPTH_FRAC,
                "SEED_LOGIT": enc.SEED_LOGIT,
                "CORRECTION_LOGIT": enc.CORRECTION_LOGIT,
                "LOGIT_CLIP": enc.LOGIT_CLIP,
                "LOW_RES": enc.LOW_RES,
                "circle_diameter_scale": 0.9,
                "circle_annotation_trust": {"full_frac": 0.5, "zero_frac": 0.95, "ramp": "cosine"},
            },
        )
        self.assertIsInstance(meta["prompt_encoding"], str)
        self.assertEqual(serve_metadata()["circle_diameter_scale"], 0.95)

    def test_metadata_comes_from_the_runs_saved_config(self) -> None:
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "config.yaml").write_text(
                "prompt:\n  edge_tau_px: 6.0\n  seed_logit: 8.0\n  circle_diameter_scale: 0.9\n", encoding="utf-8"
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                meta = serve_metadata(run_dir=run)
        self.assertEqual(meta["EDGE_TAU_PX"], 6.0)
        self.assertEqual(meta["SEED_LOGIT"], 8.0)
        self.assertEqual(meta["circle_diameter_scale"], 0.9)
        self.assertEqual(meta["CORRECTION_LOGIT"], enc.CORRECTION_LOGIT)  # not in the config: the default
        self.assertEqual(meta["EDGE_TAU_DEPTH_FRAC"], enc.EDGE_TAU_DEPTH_FRAC)
        self.assertIn("WARNING: prompt.edge_tau_px is 6.0", out.getvalue())
        self.assertIn("WARNING: prompt.seed_logit is 8.0", out.getvalue())

    def test_a_run_with_default_values_exports_without_a_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "config.yaml").write_text("prompt:\n  edge_tau_px: 8.0\n  circle_diameter_scale: 0.95\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                serve_metadata(run_dir=run)
        self.assertEqual(out.getvalue(), "")

    def test_without_a_saved_config_the_defaults_are_used_with_a_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                meta = serve_metadata(run_dir=Path(tmp))
                missing = serve_metadata()
        self.assertEqual(meta["EDGE_TAU_PX"], enc.EDGE_TAU_PX)
        self.assertEqual(missing["SEED_LOGIT"], enc.SEED_LOGIT)
        self.assertEqual(out.getvalue().count("no saved config.yaml"), 2)

    def test_the_export_reads_the_config_next_to_the_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ), contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "config.yaml").write_text("prompt:\n  correction_logit: 4.0\n", encoding="utf-8")
            out = export_serve_checkpoint(
                finetuned=run / "best_model.pt", base_ckpt=Path("b.pt"), out=Path(tmp) / "serve.pt"
            )
            payload = torch.load(out, weights_only=True)
        self.assertEqual(payload["prompt_encoding"]["CORRECTION_LOGIT"], 4.0)

    def test_the_exported_file_keeps_the_meta_layout_and_carries_the_metadata(self) -> None:
        model = nn.Linear(2, 2)
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=model
        ):
            out = export_serve_checkpoint(
                finetuned=Path("f.pt"), base_ckpt=Path("b.pt"), out=Path(tmp) / "nested" / "serve.pt",
                circle_diameter_scale=0.8,
            )
            payload = torch.load(out, weights_only=True)
        self.assertEqual(set(payload), {"model", "prompt_encoding"})
        self.assertEqual(set(payload["model"]), set(model.state_dict()))
        self.assertEqual(payload["prompt_encoding"]["circle_diameter_scale"], 0.8)
        self.assertEqual(payload["prompt_encoding"]["SEED_LOGIT"], enc.SEED_LOGIT)
        # The loader still reads this file.
        self.assertEqual(finetuned_state_dict({"model": {"image_encoder.w": 1}, "prompt_encoding": {}}), {"image_encoder.w": 1})


if __name__ == "__main__":
    unittest.main()
