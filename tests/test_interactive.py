from __future__ import annotations

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from sam2_segmentation_trainer.interactive import (
    error_clicks,
    iterative_predict_masks,
    next_click,
    simulate_clicks,
    summarize_noc,
)
from sam2_segmentation_trainer.model import decode_grouped, encode_images, predict_masks_mixed
from sam2_segmentation_trainer.train import _iterative_rounds


class NextClickTests(unittest.TestCase):
    def test_a_missed_area_gets_a_positive_click_inside_it(self) -> None:
        gt = np.zeros((20, 20), dtype=bool)
        gt[4:16, 4:16] = True
        pred = np.zeros_like(gt)
        x, y, label = next_click(pred, gt, np.random.default_rng(0))
        self.assertEqual(label, 1)
        self.assertTrue(gt[y, x])

    def test_a_spurious_area_gets_a_negative_click_inside_it(self) -> None:
        gt = np.zeros((20, 20), dtype=bool)
        pred = np.zeros_like(gt)
        pred[2:14, 2:14] = True
        x, y, label = next_click(pred, gt, np.random.default_rng(0))
        self.assertEqual(label, 0)
        self.assertTrue(pred[y, x] and not gt[y, x])

    def test_a_perfect_prediction_still_gets_a_positive_click_in_the_target(self) -> None:
        gt = np.zeros((20, 20), dtype=bool)
        gt[5:15, 5:15] = True
        x, y, label = next_click(gt.copy(), gt, np.random.default_rng(1))
        self.assertEqual(label, 1)
        self.assertTrue(gt[y, x])

    def test_error_clicks_scale_to_the_full_size_mask(self) -> None:
        gt = torch.zeros(2, 64, 64)
        gt[:, 16:48, 16:48] = 1.0
        low = torch.full((2, 16, 16), -5.0)
        coords, labels = error_clicks(low, gt, np.random.default_rng(0))
        self.assertEqual(tuple(coords.shape), (2, 2))
        self.assertTrue(bool((labels == 1).all()))
        for x, y in coords.tolist():
            self.assertTrue(16 <= x < 48 and 16 <= y < 48)


class _Encoder(nn.Module):
    """Prompt encoder stand-in that records how many points and whether a mask prompt arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.mask_downscaling = nn.Conv2d(1, 1, 2)
        self.calls: list[dict] = []

    def get_dense_pe(self) -> torch.Tensor:
        return torch.zeros(1)

    def forward(self, points, boxes, masks):
        self.calls.append(
            {
                "n_points": 0 if points is None else int((points[1] >= 0).sum(dim=1).max()),
                "has_mask": masks is not None,
            }
        )
        rows = points[0].shape[0] if points is not None else boxes.shape[0]
        count = (points[1] >= 0).sum(dim=1).float() if points is not None else torch.zeros(rows)
        sparse = count.reshape(rows, 1, 1)
        dense = torch.zeros(rows, 1, 1, 1) if masks is None else torch.full((rows, 1, 1, 1), 100.0)
        return sparse, dense


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings, **_):
        base = image_embeddings.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        value = base + sparse_prompt_embeddings.sum(dim=(1, 2)).reshape(-1, 1, 1, 1) + dense_prompt_embeddings
        return value.expand(-1, 1, 4, 4) * self.scale, torch.zeros(value.shape[0], 1), None, None


class _Sam(nn.Module):
    directly_add_no_mem_embed = False

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)
        self.sam_prompt_encoder = _Encoder()
        self.sam_mask_decoder = _Decoder()

    def forward_image(self, images):
        return images

    def _prepare_backbone_features(self, images):
        batch = images.shape[0]
        per_sample = images.mean(dim=(1, 2, 3))
        high = per_sample.reshape(1, batch, 1).expand(16, batch, 1).contiguous()
        low = per_sample.reshape(1, batch, 1).expand(4, batch, 1).contiguous()
        return None, [high, low], None, [(4, 4), (2, 2)]


def _batch(count: int = 2):
    images = torch.stack([torch.full((3, 8, 8), float(i)) for i in range(count)])
    points = torch.full((count, 1, 2), 4.0)
    labels = torch.ones(count, 1, dtype=torch.int32)
    gt = torch.zeros(count, 8, 8, dtype=torch.long)
    gt[:, 2:6, 2:6] = 1
    return images, points, labels, gt


class IterativePredictTests(unittest.TestCase):
    def test_rounds_add_clicks_and_feed_the_previous_mask(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        pred, _iou = iterative_predict_masks(
            model, images, points, labels, [None, None], [None, None], gt,
            image_size=8, rounds=3, prev_mask_prob=1.0, rng=np.random.default_rng(0),
        )
        self.assertEqual(tuple(pred.shape), (2, 8, 8))
        calls = model.sam_prompt_encoder.calls
        self.assertEqual([c["n_points"] for c in calls], [1, 2, 3, 4])
        self.assertEqual([c["has_mask"] for c in calls], [False, True, True, True])

    def test_zero_rounds_matches_the_plain_mixed_prediction(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        plain, _ = predict_masks_mixed(model, images, points, labels, [None, None], [None, None], image_size=8)
        same, _ = iterative_predict_masks(
            model, images, points, labels, [None, None], [None, None], gt,
            image_size=8, rounds=0, prev_mask_prob=0.5, rng=np.random.default_rng(0),
        )
        self.assertTrue(torch.allclose(plain, same))

    def test_only_the_last_pass_carries_gradients(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        pred, _ = iterative_predict_masks(
            model, images, points, labels, [None, None], [None, None], gt,
            image_size=8, rounds=2, prev_mask_prob=1.0, rng=np.random.default_rng(0),
        )
        self.assertTrue(pred.requires_grad)
        pred.sum().backward()
        self.assertIsNotNone(model.sam_mask_decoder.scale.grad)

    def test_prev_mask_probability_zero_keeps_the_initial_prompt(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        iterative_predict_masks(
            model, images, points, labels, [None, None], [None, None], gt,
            image_size=8, rounds=2, prev_mask_prob=0.0, rng=np.random.default_rng(0),
        )
        self.assertFalse(any(c["has_mask"] for c in model.sam_prompt_encoder.calls))

    def test_decode_grouped_returns_low_res_when_asked(self) -> None:
        model = _Sam()
        images, points, labels, _gt = _batch()
        embed, high = encode_images(model, images)
        out = decode_grouped(
            model, embed, high, points, labels, [None, None], [None, None],
            image_size=8, return_low_res=True,
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[2].shape), (2, 4, 4))


class SimulateClicksTests(unittest.TestCase):
    def test_reports_clicks_and_failures(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        batch = {"image": images, "mask": gt, "point": points, "point_label": labels}
        rows = simulate_clicks(model, batch, "cpu", 8, max_clicks=4, target_iou=1.1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["noc"] == 4 and not r["reached"] for r in rows))
        easy = simulate_clicks(model, batch, "cpu", 8, max_clicks=4, target_iou=0.0)
        self.assertTrue(all(r["noc"] == 1 and r["reached"] for r in easy))

    def test_summary_groups_by_volume(self) -> None:
        rows = [
            {"noc": 2, "reached": True, "iou": 0.9},
            {"noc": 4, "reached": False, "iou": 0.5},
            {"noc": 6, "reached": True, "iou": 0.9},
        ]
        summary = summarize_noc(rows, ["RC1", "RC1", "RC2"], 20)
        self.assertAlmostEqual(summary["overall"]["mean_clicks"], 4.0)
        self.assertAlmostEqual(summary["by_volume"]["RC1"]["failed"], 0.5)
        self.assertEqual(summarize_noc([], [], 20), {"n": 0})


class RoundsDrawTests(unittest.TestCase):
    def test_off_by_default_and_repeatable(self) -> None:
        self.assertEqual(_iterative_rounds(OmegaConf.create({}), 1, 0, 0), 0)
        cfg = OmegaConf.create({"iterative": {"batch_fraction": 1.0, "rounds_min": 2, "rounds_max": 4}})
        draws = {_iterative_rounds(cfg, 1, 0, step) for step in range(60)}
        self.assertEqual(draws, {2, 3, 4})
        self.assertEqual(_iterative_rounds(cfg, 1, 3, 7), _iterative_rounds(cfg, 1, 3, 7))

    def test_fraction_controls_the_share(self) -> None:
        cfg = OmegaConf.create({"iterative": {"batch_fraction": 0.2}})
        hits = sum(_iterative_rounds(cfg, 5, 0, step) > 0 for step in range(2000))
        self.assertGreater(hits, 300)
        self.assertLess(hits, 500)


if __name__ == "__main__":
    unittest.main()
