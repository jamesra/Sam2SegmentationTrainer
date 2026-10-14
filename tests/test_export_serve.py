from __future__ import annotations

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
            },
        )
        self.assertIsInstance(meta["prompt_encoding"], str)
        self.assertEqual(serve_metadata()["circle_diameter_scale"], 0.95)

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
