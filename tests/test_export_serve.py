from __future__ import annotations

import unittest

from sam2_segmentation_trainer.export_serve import finetuned_state_dict


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


if __name__ == "__main__":
    unittest.main()
