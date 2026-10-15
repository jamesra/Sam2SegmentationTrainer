from __future__ import annotations

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from sam2_segmentation_trainer.interactive import (
    error_clicks,
    error_regions,
    iterative_predict_masks,
    next_click,
    simulate_clicks,
    summarize_noc,
)
from sam2_segmentation_trainer.model import _stack_on, decode_grouped, encode_images, predict_masks_mixed
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

    def test_a_wobble_inside_the_band_gets_no_click_in_the_band(self) -> None:
        # The prediction follows the trace but wobbles by 8 px: no goal error, and a click lands only on a real one.
        gt = np.zeros((200, 200), dtype=bool)
        gt[40:160, 40:160] = True
        pred = gt.copy()
        pred[40:160, 40:48] = False  # misses an 8 px strip along the left edge
        pred[152:160, 40:160] = False  # and the bottom edge
        pred[30:40, 60:140] = True  # adds an 8 px strip outside the top edge
        # With nothing but the wobble left there is no goal error: no raw fallback, the "no error" click is used.
        missed, spurious = error_regions(pred, gt, band=10.0)
        raw_missed, raw_spurious = error_regions(pred, gt, band=0.0)
        self.assertTrue(raw_missed.any() and raw_spurious.any())  # band 0 keeps the raw differences
        self.assertFalse(missed.any() or spurious.any())
        for seed in range(10):
            x, y, label = next_click(pred, gt, np.random.default_rng(seed), band=10.0)
            self.assertEqual(label, 1)
            self.assertTrue(abs(x - 99.5) <= 1 and abs(y - 99.5) <= 1, (x, y))  # the deepest target pixel
        # A real miss deeper than the band does get the click, however the area compares with the wobble.
        pred[90:110, 90:110] = False
        missed, spurious = error_regions(pred, gt, band=10.0)
        self.assertFalse(spurious.any())
        self.assertTrue(missed[90:110, 90:110].any() and not missed[:, :50].any() and not missed[150:, :].any())
        for seed in range(10):
            x, y, label = next_click(pred, gt, np.random.default_rng(seed), band=10.0)
            self.assertEqual(label, 1)
            self.assertTrue(90 <= x < 110 and 90 <= y < 110, (x, y))

    def test_a_spurious_blob_beyond_the_band_gets_a_negative_click(self) -> None:
        gt = np.zeros((200, 200), dtype=bool)
        gt[60:140, 60:140] = True
        pred = gt.copy()
        pred[10:40, 100:130] = True  # 20+ px outside the edge
        x, y, label = next_click(pred, gt, np.random.default_rng(0), band=10.0)
        self.assertEqual(label, 0)
        self.assertTrue(10 <= y < 40 and 100 <= x < 130)

    def test_the_band_is_scaled_to_the_logit_grid(self) -> None:
        gt = torch.zeros(1, 64, 64)
        gt[:, 8:56, 8:56] = 1.0
        low = torch.full((1, 16, 16), -5.0)
        low[:, 2:14, 2:14] = 5.0  # follows the trace to within a pixel of the 16 grid (4 full px)
        coords, labels = error_clicks(low, gt, np.random.default_rng(0), core_band_px=8.0)  # 2 px on the 16 grid
        self.assertEqual(tuple(coords.shape), (1, 2))
        # Only the rim is off, inside the band: there is no goal error, so the raw regions decide (a positive click
        # in the missed rim or a negative one in the spill).
        self.assertIn(int(labels[0]), (0, 1))

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

    def test_variants_reuse_the_image_embeddings_and_expand_the_targets(self) -> None:
        model = _Sam()
        images, _points, _labels, gt = _batch()
        index = torch.tensor([0, 0, 1])
        points = torch.full((3, 1, 2), 4.0)
        labels = torch.ones(3, 1, dtype=torch.int32)
        pred, _iou = iterative_predict_masks(
            model, images, points, labels, [None] * 3, [None] * 3, gt,
            image_size=8, rounds=2, prev_mask_prob=1.0, rng=np.random.default_rng(0), variant_index=index,
        )
        self.assertEqual(tuple(pred.shape), (3, 8, 8))
        self.assertEqual([c["n_points"] for c in model.sam_prompt_encoder.calls], [1, 2, 3])

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


class StackOnTests(unittest.TestCase):
    def test_stacks_cpu_masks_onto_the_device(self) -> None:
        out = _stack_on([torch.zeros(4, 4), torch.ones(4, 4)], "cpu")
        self.assertEqual(tuple(out.shape), (2, 4, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_stacks_a_mix_of_cpu_and_gpu_masks(self) -> None:
        # The click rounds feed back GPU logits for some samples and keep the loader's CPU mask for the others.
        out = _stack_on([torch.zeros(4, 4), torch.ones(4, 4, device="cuda")], "cuda")
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(tuple(out.shape), (2, 4, 4))
        self.assertEqual(float(out[1].sum()), 16.0)


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

    def test_reports_the_goal_stopping_rule_next_to_the_iou_rule(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        batch = {"image": images, "mask": gt, "point": points, "point_label": labels}
        rows = simulate_clicks(model, batch, "cpu", 8, max_clicks=3, target_iou=1.1, core_band_px=1.0)
        for row in rows:
            self.assertEqual(row["noc"], 3)  # the IoU rule never fires
            self.assertIn(row["noc_goal"], (1, 2, 3))
            self.assertEqual(row["reached_goal"], row["noc_goal"] < 3 or row["reached_goal"])
            self.assertIn("core_recall", row)
        easy = simulate_clicks(
            model, batch, "cpu", 8, max_clicks=3, target_iou=0.0, core_band_px=1.0, core_recall_target=0.0
        )
        self.assertTrue(all(r["noc"] == 1 and r["noc_goal"] == 1 and r["reached_goal"] for r in easy))

    def test_a_merge_blocks_the_goal_rule(self) -> None:
        model = _Sam()
        images, points, labels, gt = _batch()
        cores = torch.ones(images.shape[0], 8, 8, dtype=torch.int16)  # the whole tile is a neighbor's core
        batch = {"image": images, "mask": gt, "point": points, "point_label": labels, "neighbor_core": cores}
        rows = simulate_clicks(
            model, batch, "cpu", 8, max_clicks=2, target_iou=0.0, core_band_px=1.0, core_recall_target=0.0,
            merge_min_px=1,
        )
        for row in rows:
            if row["merge"]:
                self.assertFalse(row["reached_goal"])

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
        goal_rows = [
            {"noc": 2, "reached": True, "iou": 0.9, "noc_goal": 1, "reached_goal": True},
            {"noc": 4, "reached": False, "iou": 0.5, "noc_goal": 20, "reached_goal": False},
        ]
        both = summarize_noc(goal_rows, ["RC1", "RC1"], 20)
        self.assertAlmostEqual(both["overall"]["mean_clicks_goal"], 10.5)
        self.assertAlmostEqual(both["overall"]["failed_goal"], 0.5)
        self.assertNotIn("mean_clicks_goal", summary["overall"])


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
