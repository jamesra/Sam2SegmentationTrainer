from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from annotation_crops.negatives import CIRCLE_FILE_NAME
from sam2_segmentation_trainer.data.dataset import EMSegDataset, EpochCounter, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import FULL_TILE_PROMPTS, PromptConfig, PromptType
from sam2_segmentation_trainer.data.tiles import TileContext, build_tile_context
from sam2_segmentation_trainer.losses import (
    containment_warmup_scale,
    edge_loss_weight,
    mask_containment_loss_per_sample,
    tolerant_iou_target,
    weighted_dice_loss,
    weighted_focal_loss_per_sample,
)
from sam2_segmentation_trainer.paths import annotation_crops_root
from sam2_segmentation_trainer.model import (
    combined_loss,
    configure_trainable_params,
    param_groups,
    predict_masks_mixed,
)

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops

_ZERO = {
    "prob_points_only": 0.0,
    "prob_box_only": 0.0,
    "prob_box_points": 0.0,
    "prob_circle_seed": 0.0,
    "prob_circle_annotator": 0.0,
    "prob_circle_pull_through": 0.0,
    "prob_adjacent_poly_seed": 0.0,
    "prob_subset_seed": 0.0,
    "prob_correction": 0.0,
    "prob_near_correct": 0.0,
}


def _cfg(**kwargs: float) -> PromptConfig:
    """One prompt family at probability 1 (pass it as ``prob_...=1.0``) and a single variant unless asked."""
    values = dict(_ZERO)
    values["variants_per_sample"] = 1
    values.update(kwargs)
    return PromptConfig(**values)


def _write_circles(root: Path, volume: str, cells: dict) -> None:
    path = annotation_crops_root(volume, root) / CIRCLE_FILE_NAME
    path.write_text(json.dumps({"version": 1, "diameter_scale": 0.8, "grid_cells": cells}), encoding="utf-8")


class MixedDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=6)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        # Tile 0 has locations 1 (big square) and 2 (small square at 2:6).
        self.shared = [ex for ex in self.examples if ex.image_key.startswith("RC1_0_")]
        self.alone = [ex for ex in self.examples if ex.image_key.startswith("RC1_1_")]
        self.assertEqual(len(self.shared), 2)
        self.key = self.shared[0].image_key
        self.alone_key = self.alone[0].image_key

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dataset(self, cfg: PromptConfig, examples=None, epoch: EpochCounter | None = None, **extra) -> EMSegDataset:
        examples = self.examples if examples is None else examples
        return EMSegDataset(
            examples,
            split="val",
            image_size=32,
            tiles=build_tile_context(self.examples),
            prompt_cfg=cfg,
            seed=7,
            epoch=epoch,
            **extra,
        )

    def _item(self, ds: EMSegDataset, location_id: int, key: str | None = None) -> dict:
        key = key or self.key
        for index, ex in enumerate(ds.examples):
            if ex.location_id == location_id and ex.image_key == key:
                return ds[index]
        raise AssertionError("example not found")

    def test_legacy_item_has_the_new_keys_unset(self) -> None:
        ds = EMSegDataset(self.examples, split="val", image_size=32)
        item = ds[0]
        (variant,) = item["variants"]
        self.assertEqual(variant["prompt_type"], "centroid")
        self.assertIsNone(variant["boxes"])
        self.assertIsNone(variant["dense_mask"])
        batch = collate_em([ds[0], ds[1]])
        self.assertEqual(tuple(batch["point"].shape), (2, 1, 2))
        self.assertEqual(tuple(batch["prompt_fg"].shape), (2, 32, 32))
        self.assertEqual(batch["prompt_type"], ["centroid", "centroid"])
        self.assertIsNone(batch["prompt_input"])

    def test_neighbor_becomes_negative_in_a_circle_seed_prompt(self) -> None:
        loc = self.shared[0].location_id
        _write_circles(
            self.root,
            "RC1",
            {self.key: [{"location_id": loc, "center_x": 16.0, "center_y": 16.0, "radius": 4.0,
                         "has_polygon_mask": True, "source": "inscribed"}]},
        )
        ds = self._dataset(_cfg(prob_circle_seed=1.0))
        item = self._item(ds, loc)
        (variant,) = item["variants"]
        self.assertEqual(variant["prompt_type"], "circle_seed")
        dense = variant["dense_mask"]
        self.assertEqual(tuple(dense.shape), (256, 256))
        self.assertLess(float(dense[32, 32]), -3.0)  # neighbor tile pixels 2:6 -> low-res rows 16:48; a 4 px square is all edge
        self.assertGreater(float(dense[128, 128]), 6.0)  # seed core of a small circle
        self.assertGreater(int(variant["prompt_fg"].gt(0).sum()), 20)
        self.assertIsNone(variant["boxes"])

    def test_tile_without_neighbors_uses_no_mask_embedding(self) -> None:
        # No family is forced on a sample, so points at probability 1 is the only variant.
        ds = self._dataset(_cfg(prob_points_only=1.0, variants_per_sample=2))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        (only,) = item["variants"]  # nothing else has a share: the plan stops after it
        self.assertEqual(only["prompt_type"], "points")
        self.assertIsNone(only["dense_mask"])
        self.assertTrue((only["point_label"] == 1).all())

    def test_points_prompt_adds_background_points_in_the_neighbor(self) -> None:
        ds = self._dataset(_cfg(prob_points_only=1.0, bg_point_prob_polygon=1.0))
        item = self._item(ds, self.shared[0].location_id)
        points = item["variants"][0]
        labels = points["point_label"]
        self.assertTrue((labels == 0).any() and (labels == 1).any())
        for (x, y), label in zip(points["point"].tolist(), labels.tolist()):
            if label == 0:
                self.assertTrue(2 <= x < 6 and 2 <= y < 6)

    def test_box_prompt_contains_the_object(self) -> None:
        ds = self._dataset(_cfg(prob_box_only=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        variant = item["variants"][0]
        box = variant["boxes"].tolist()
        self.assertTrue(box[0] <= 8 and box[1] <= 8 and box[2] >= 24 and box[3] >= 24)
        self.assertEqual(variant["point"].shape[0], 0)

    def test_circle_seed_prompt_uses_the_annotation_circle(self) -> None:
        loc = self.alone[0].location_id
        _write_circles(
            self.root,
            "RC1",
            {self.alone_key: [{"location_id": loc, "center_x": 16.0, "center_y": 16.0, "radius": 4.0,
                               "has_polygon_mask": True, "source": "inscribed"}]},
        )
        ds = self._dataset(_cfg(prob_circle_seed=1.0))
        item = self._item(ds, loc, self.alone_key)
        (variant,) = item["variants"]
        self.assertEqual(variant["prompt_type"], "circle_seed")
        self.assertEqual(variant["point"].shape, (1, 2))
        self.assertAlmostEqual(float(variant["point"][0, 0]), 15.5, places=1)
        self.assertGreater(int(variant["prompt_input"].sum()), 20)
        self.assertLess(int(variant["prompt_input"].sum()), int(item["mask"].sum()))
        self.assertGreater(float(variant["dense_mask"][128, 128]), 6.0)

    def test_circle_seed_fits_a_circle_when_the_tile_has_none(self) -> None:
        ds = self._dataset(_cfg(prob_circle_seed=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        (variant,) = item["variants"]
        self.assertEqual(variant["prompt_type"], "circle_seed")
        self.assertGreater(int(variant["prompt_input"].sum()), 20)
        self.assertLess(int(variant["prompt_input"].sum()), int(item["mask"].sum()))

    def test_unmasked_circle_in_the_tile_is_a_negative(self) -> None:
        _write_circles(
            self.root,
            "RC1",
            {self.alone_key: [{"location_id": 9999, "center_x": 4.0, "center_y": 28.0, "radius": 3.0}]},
        )
        ds = self._dataset(_cfg(prob_circle_seed=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        dense = item["variants"][0]["dense_mask"]
        self.assertLess(float(dense[int(28 / 32 * 256), int(4 / 32 * 256)]), -4.0)

    def test_prompts_replay_for_the_same_epoch_index_and_seed(self) -> None:
        counter = EpochCounter()
        ds = self._dataset(_cfg(prob_points_only=1.0, bg_point_prob_polygon=1.0, variants_per_sample=3), epoch=counter)
        counter.set(3)
        first = self._item(ds, self.shared[0].location_id)
        second = self._item(ds, self.shared[0].location_id)
        self.assertEqual(len(first["variants"]), len(second["variants"]))
        for a, b in zip(first["variants"], second["variants"]):
            self.assertEqual(a["prompt_type"], b["prompt_type"])
            self.assertTrue(torch.equal(a["point"], b["point"]))
        self.assertEqual(counter.get(), 3)

    def test_variants_share_one_image_and_draw_without_replacement(self) -> None:
        ds = self._dataset(PromptConfig())
        for index in range(len(ds)):
            requested = [v["requested_prompt_type"] for v in ds[index]["variants"]]
            self.assertEqual(len(requested), 4)
            self.assertEqual(len(set(requested)), 4)

    def test_an_adjacent_poly_seed_is_drawn_only_when_a_shape_exists_at_any_position(self) -> None:
        cfg = PromptConfig()
        ds = self._dataset(cfg)
        none = [ds._plan_types(cfg, i, 0, full_tile=False, forced=None) for i in range(300)]
        self.assertFalse(any(PromptType.ADJACENT_POLY_SEED in plan for plan in none))
        with_shape = [ds._plan_types(cfg, i, 0, full_tile=False, forced=None, has_adjacent=True) for i in range(600)]
        positions = {plan.index(PromptType.ADJACENT_POLY_SEED) for plan in with_shape if PromptType.ADJACENT_POLY_SEED in plan}
        self.assertEqual(positions, {0, 1, 2, 3}, "no position is reserved for any family")
        first = sum(plan[0] is PromptType.ADJACENT_POLY_SEED for plan in with_shape) / len(with_shape)
        self.assertAlmostEqual(first, cfg.prob_adjacent_poly_seed, delta=0.05)  # the first draw is the plain mix
        for plan in with_shape:
            self.assertEqual(len(set(plan)), len(plan))
        full = [ds._plan_types(cfg, i, 0, full_tile=True, forced=None, has_adjacent=True) for i in range(50)]
        for plan in full:
            self.assertNotIn(PromptType.ADJACENT_POLY_SEED, plan)
            # Points, circle_seed and circle_annotator: no pull-through on a full tile, so 3 of the 4 variants.
            self.assertEqual(len(plan), 3)
            self.assertEqual(set(plan), set(FULL_TILE_PROMPTS))
            self.assertNotIn(PromptType.CIRCLE_PULL_THROUGH, plan)

    def test_a_stub_shape_makes_an_adjacent_poly_seed_item(self) -> None:
        class StubTiles(TileContext):
            def adjacent_shape(self, key, location_id, **_kwargs):
                shape = np.zeros((32, 32), dtype=bool)
                shape[11:21, 11:21] = True
                return shape

        base = build_tile_context(self.examples)
        ds = EMSegDataset(
            self.examples, split="val", image_size=32, seed=7,
            tiles=StubTiles(neighbors=base.neighbors, circles=base.circles),
            prompt_cfg=_cfg(prob_adjacent_poly_seed=1.0),
        )
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        (variant,) = item["variants"]
        self.assertEqual(variant["prompt_type"], "adjacent_poly_seed")
        self.assertIsNotNone(variant["prompt_fg"])

    def test_leak_weight_multiplies_the_loss_weight_on_neighbor_cores(self) -> None:
        from annotation_crops.negatives import KIND_POLYGON, NegativeRegion
        from sam2_segmentation_trainer.data.meta import SampleMeta

        ds = self._dataset(_cfg(prob_points_only=1.0), edge_tau_px=8.0, leak_weight=3.0, leak_core_px=10.0)
        target = np.zeros((200, 200), dtype=bool)
        target[40:120, 40:120] = True
        neighbor = np.zeros((200, 200), dtype=bool)
        neighbor[30:190, 130:195] = True
        meta = SampleMeta("RC1", "k", 1, 1, "", "k")
        plain = ds._weights(target, [], meta)["edge_weight"].float()
        weighted = ds._weights(target, [NegativeRegion(2, KIND_POLYGON, neighbor)], meta)["edge_weight"].float()
        self.assertAlmostEqual(float(weighted[100, 160]), 3.0, places=2)  # deep in the neighbor, far from the target
        self.assertAlmostEqual(float(plain[100, 160]), 1.0, places=2)
        self.assertAlmostEqual(float(weighted[100, 80]), float(plain[100, 80]), places=3)  # inside the target: same
        self.assertAlmostEqual(float(weighted[185, 20]), 1.0, places=3)  # nowhere near anything
        self.assertEqual(weighted.dtype, torch.float32)
        self.assertEqual(ds._weights(target, [], meta)["edge_weight"].dtype, torch.float16)
        # The rim of the neighbor (5 px inside) is not its core.
        self.assertAlmostEqual(float(weighted[100, 135]), float(plain[100, 135]), places=3)

    def test_leak_weight_works_with_the_edge_weight_off(self) -> None:
        from annotation_crops.negatives import KIND_POLYGON, NegativeRegion
        from sam2_segmentation_trainer.data.meta import SampleMeta

        ds = self._dataset(_cfg(prob_points_only=1.0), edge_tau_px=0.0, leak_weight=3.0)
        target = np.zeros((200, 200), dtype=bool)
        target[40:80, 40:80] = True
        neighbor = np.zeros((200, 200), dtype=bool)
        neighbor[100:190, 100:190] = True
        meta = SampleMeta("RC1", "k", 1, 1, "", "k")
        weight = ds._weights(target, [NegativeRegion(2, KIND_POLYGON, neighbor)], meta)["edge_weight"].float()
        self.assertAlmostEqual(float(weight[150, 150]), 3.0, places=2)
        self.assertAlmostEqual(float(weight[60, 60]), 1.0, places=3)
        self.assertIsNone(ds._weights(target, [], meta)["edge_weight"])

    def test_approved_items_are_flagged_and_weighted(self) -> None:
        first = self.shared[0]
        ds = self._dataset(
            _cfg(prob_circle_seed=1.0), approved={(first.location_id, first.image_key)}, approved_weight=2.0
        )
        item = self._item(ds, first.location_id)
        self.assertEqual(item["sample_weight"], 2.0)
        self.assertTrue(item["sample_meta"].approved)
        other = self._item(ds, self.shared[1].location_id)
        self.assertEqual(other["sample_weight"], 1.0)
        self.assertFalse(other["sample_meta"].approved)
        batch = collate_em([item, other])
        self.assertEqual(batch["sample_weight"].tolist(), [2.0, 1.0])
        self.assertTrue(batch["meta"][0]["approved"])

    def test_neighbor_cores_come_with_a_validation_band(self) -> None:
        for mode in ("point", "circle_mask"):
            ds = self._dataset(_cfg(prob_circle_seed=1.0), core_band_px=1.0)
            ds.val_prompt = mode
            item = self._item(ds, self.shared[0].location_id)
            cores = item["neighbor_core"]
            self.assertEqual(tuple(cores.shape), (32, 32))
            self.assertEqual(cores.dtype, torch.int16)
            self.assertGreater(int(cores.max()), 0)  # the 4 px neighbor has a 1 px core
            batch = collate_em([item])
            self.assertEqual(tuple(batch["neighbor_core"].shape), (1, 32, 32))
        off = self._dataset(_cfg(prob_circle_seed=1.0))
        self.assertIsNone(self._item(off, self.shared[0].location_id)["neighbor_core"])
        self.assertIsNone(collate_em([self._item(off, self.shared[0].location_id)])["neighbor_core"])

    def test_collate_expands_variants_and_repeats_meta(self) -> None:
        ds = self._dataset(PromptConfig(variants_per_sample=3))
        batch = collate_em([ds[0], ds[1]])
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 32, 32))
        self.assertEqual(tuple(batch["mask"].shape), (2, 32, 32))
        self.assertEqual(batch["variant_index"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(batch["point"].shape[0], 6)
        self.assertEqual(len(batch["dense_mask"]), 6)
        self.assertEqual(tuple(batch["prompt_fg"].shape), (6, 32, 32))
        self.assertEqual(len(batch["meta"]), 6)
        self.assertEqual(len(batch["prompt_type"]), 6)
        self.assertEqual(batch["meta"][0], batch["meta"][2])

    def test_edge_weight_follows_the_distance_to_the_mask_edge(self) -> None:
        ds = self._dataset(_cfg(prob_circle_seed=1.0), edge_tau_px=2.5, edge_weight_min=0.05)
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        weight = item["edge_weight"].float()
        self.assertEqual(tuple(weight.shape), (32, 32))
        self.assertGreater(float(weight[16, 16]), 0.94)  # 8 px from the edge
        self.assertLess(float(weight[16, 8]), 0.35)  # first pixel of the mask: half a pixel from the edge
        self.assertLess(float(weight[16, 7]), 0.35)  # first pixel outside
        self.assertGreater(float(weight[16, 3]), 0.7)  # well outside
        self.assertGreaterEqual(float(weight.min()), 0.05 - 1e-3)
        self.assertIsNone(self._item(self._dataset(_cfg(prob_circle_seed=1.0)), self.alone[0].location_id, self.alone_key)["edge_weight"])

    def test_collate_pads_points_and_keeps_lists(self) -> None:
        def item(points: int, box: bool, dense: bool) -> dict:
            from sam2_segmentation_trainer.data.meta import SampleMeta

            variant = {
                "point": torch.arange(points * 2, dtype=torch.float32).reshape(points, 2),
                "point_label": torch.ones(points, dtype=torch.int32),
                "boxes": torch.zeros(4) if box else None,
                "dense_mask": torch.zeros(256, 256) if dense else None,
                "prompt_fg": torch.ones(4, 4, dtype=torch.float16) if dense else None,
                "prompt_input": None,
                "prompt_type": "x",
            }
            return {
                "image": torch.zeros(3, 4, 4),
                "mask": torch.zeros(4, 4, dtype=torch.int64),
                "edge_weight": None,
                "variants": [variant],
                "orig_hw": (4, 4),
                "sample_meta": SampleMeta("V", "K", 1, 1, "c", "V:1"),
            }

        batch = collate_em([item(1, False, False), item(3, True, False), item(0, True, True)])
        self.assertEqual(tuple(batch["point"].shape), (3, 3, 2))
        self.assertEqual(batch["point_label"].tolist(), [[1, -1, -1], [1, 1, 1], [-1, -1, -1]])
        self.assertEqual([b is not None for b in batch["boxes"]], [False, True, True])
        self.assertEqual([d is not None for d in batch["dense_mask"]], [False, False, True])
        self.assertEqual(batch["prompt_fg"].float().sum(dim=(1, 2)).tolist(), [0, 0, 16])
        self.assertEqual(batch["variant_index"].tolist(), [0, 1, 2])


class _FakeEncoder(nn.Module):
    """Prompt encoder stand-in: reports what each sample's prompt contained as one number."""

    def __init__(self) -> None:
        super().__init__()
        self.mask_downscaling = nn.Conv2d(1, 1, 2)
        self.point_embed = nn.Embedding(2, 2)
        self.calls: list[dict] = []

    def get_dense_pe(self) -> torch.Tensor:
        return torch.zeros(1)

    def forward(self, points, boxes, masks):
        self.calls.append({"points": points, "boxes": boxes, "masks": masks})
        if points is not None:
            count = (points[1] >= 0).sum(dim=1).float()
        else:
            count = torch.zeros(boxes.shape[0])
        rows = count.shape[0]
        sparse = count.reshape(rows, 1, 1)
        if boxes is not None:
            sparse = sparse + 10.0
        dense = torch.zeros(rows, 1, 1, 1) if masks is None else torch.full((rows, 1, 1, 1), 100.0)
        return sparse, dense


class _FakeDecoder(nn.Module):
    def forward(self, *, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings, **_):
        base = image_embeddings.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        value = base + sparse_prompt_embeddings.sum(dim=(1, 2)).reshape(-1, 1, 1, 1) + dense_prompt_embeddings
        return value.expand(-1, 1, 4, 4), torch.zeros(value.shape[0], 1), None, None


class _FakeSam(nn.Module):
    directly_add_no_mem_embed = False

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)
        self.sam_prompt_encoder = _FakeEncoder()
        self.sam_mask_decoder = _FakeDecoder()

    def forward_image(self, images):
        return images

    def _prepare_backbone_features(self, images):
        batch = images.shape[0]
        per_sample = images.mean(dim=(1, 2, 3))
        high = per_sample.reshape(1, batch, 1).expand(16, batch, 1).contiguous()
        low = per_sample.reshape(1, batch, 1).expand(4, batch, 1).contiguous()
        return None, [high, low], None, [(4, 4), (2, 2)]


class PredictMasksMixedTests(unittest.TestCase):
    def test_groups_by_prompt_structure_and_restores_batch_order(self) -> None:
        model = _FakeSam()
        images = torch.stack([torch.full((3, 4, 4), float(i)) for i in range(4)])
        points = torch.zeros(4, 3, 2)
        labels = torch.tensor([[1, -1, -1], [1, 1, -1], [1, -1, -1], [1, 1, 1]], dtype=torch.int32)
        points[:, :, :] = 2.0
        boxes = [None, torch.tensor([1.0, 2.0, 3.0, 4.0]), None, None]
        dense = [None, None, torch.zeros(256, 256), None]
        logits, iou = predict_masks_mixed(
            model, images, points, labels, boxes, dense, image_size=8, orig_hw=(4, 4)
        )
        self.assertEqual(tuple(logits.shape), (4, 8, 8))
        self.assertEqual(tuple(iou.shape), (4, 1))
        values = logits[:, 0, 0].tolist()
        # image value + valid points + 10 if box + 100 if dense mask
        self.assertEqual(values, [0 + 1, 1 + 2 + 10, 2 + 1 + 100, 3 + 3])

        calls = model.sam_prompt_encoder.calls
        self.assertEqual(len(calls), 3)  # (no box, no mask) x2 samples, box, mask
        box_call = next(c for c in calls if c["boxes"] is not None)
        self.assertEqual(box_call["boxes"].tolist(), [[2.0, 4.0, 6.0, 8.0]])  # 4px tile -> 8px model
        mask_call = next(c for c in calls if c["masks"] is not None)
        self.assertEqual(tuple(mask_call["masks"].shape), (1, 1, 256, 256))
        plain = [c for c in calls if c["boxes"] is None and c["masks"] is None]
        self.assertEqual(len(plain), 1)
        self.assertEqual(plain[0]["points"][1].tolist(), [[1, -1, -1], [1, 1, 1]])

    def test_single_group_is_one_decode(self) -> None:
        model = _FakeSam()
        images = torch.zeros(2, 3, 4, 4)
        points = torch.zeros(2, 1, 2)
        labels = torch.ones(2, 1, dtype=torch.int32)
        logits, _ = predict_masks_mixed(
            model, images, points, labels, [None, None], [None, None], image_size=4
        )
        self.assertEqual(tuple(logits.shape), (2, 4, 4))
        self.assertEqual(len(model.sam_prompt_encoder.calls), 1)

    def test_box_only_batch_passes_no_points(self) -> None:
        model = _FakeSam()
        images = torch.zeros(2, 3, 4, 4)
        points = torch.zeros(2, 0, 2)
        labels = torch.zeros(2, 0, dtype=torch.int32)
        boxes = [torch.tensor([0.0, 0.0, 2.0, 2.0])] * 2
        logits, _ = predict_masks_mixed(
            model, images, points, labels, boxes, [None, None], image_size=4
        )
        self.assertIsNone(model.sam_prompt_encoder.calls[0]["points"])
        self.assertEqual(logits[:, 0, 0].tolist(), [10.0, 10.0])

    def test_gradient_reaches_every_group(self) -> None:
        model = _FakeSam()
        images = torch.zeros(2, 3, 4, 4, requires_grad=True)
        points = torch.ones(2, 1, 2)
        labels = torch.ones(2, 1, dtype=torch.int32)
        boxes = [None, torch.tensor([0.0, 0.0, 1.0, 1.0])]
        logits, _ = predict_masks_mixed(
            model, images, points, labels, boxes, [None, None], image_size=4
        )
        logits.sum().backward()
        self.assertIsNotNone(images.grad)


class TrainableParamsTests(unittest.TestCase):
    def test_mask_cnn_is_the_only_frozen_prompt_part_that_can_thaw(self) -> None:
        model = _FakeSam()
        configure_trainable_params(model, "none")
        self.assertFalse(any(p.requires_grad for p in model.sam_prompt_encoder.parameters()))
        configure_trainable_params(model, "none", unfreeze_mask_cnn=True)
        self.assertTrue(all(p.requires_grad for p in model.sam_prompt_encoder.mask_downscaling.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.sam_prompt_encoder.point_embed.parameters()))
        groups = param_groups(model, 1e-4, 0.1)
        other_ids = {id(p) for p in groups[0]["params"]}
        self.assertTrue(
            all(id(p) in other_ids for p in model.sam_prompt_encoder.mask_downscaling.parameters())
        )


class MaskContainmentLossTests(unittest.TestCase):
    def test_zero_when_prompted_pixels_are_foreground(self) -> None:
        pred = torch.full((2, 4, 4), 3.0)
        fg = torch.zeros(2, 4, 4, dtype=torch.uint8)
        fg[:, :2] = 1
        self.assertTrue(torch.equal(mask_containment_loss_per_sample(pred, fg), torch.zeros(2)))

    def test_penalizes_background_only_under_the_mask_and_ignores_samples_without_one(self) -> None:
        pred = torch.full((2, 4, 4), -2.0)
        fg = torch.zeros(2, 4, 4, dtype=torch.uint8)
        fg[0, :2] = 1  # sample 1 has no mask prompt
        loss = mask_containment_loss_per_sample(pred, fg)
        self.assertAlmostEqual(float(loss[0]), 2.0)
        self.assertEqual(float(loss[1]), 0.0)

    def test_free_to_grow_beyond_the_mask(self) -> None:
        pred = torch.full((1, 4, 4), 5.0)
        fg = torch.zeros(1, 4, 4, dtype=torch.uint8)
        fg[:, 1, 1] = 1
        self.assertEqual(float(mask_containment_loss_per_sample(pred, fg)[0]), 0.0)

    def test_combined_loss_adds_the_term_only_when_weighted(self) -> None:
        pred = torch.full((2, 8, 8), -1.0, requires_grad=True)
        target = torch.zeros(2, 8, 8, dtype=torch.int64)
        iou = torch.zeros(2, 1)
        fg = torch.zeros(2, 8, 8, dtype=torch.uint8)
        fg[:, :4] = 1
        kwargs = dict(focal_weight=1.0, dice_weight=1.0, iou_weight=1.0)
        base, base_stats = combined_loss(pred, target, iou, **kwargs)
        self.assertNotIn("loss_mask_contain", base_stats)
        loss, stats = combined_loss(
            pred, target, iou, prompt_foreground=fg, mask_contain_weight=0.5, **kwargs
        )
        self.assertAlmostEqual(float(stats["loss_mask_contain"]), 1.0, places=5)
        self.assertAlmostEqual(float((loss - base).detach()), 0.5, places=5)
        same, same_stats = combined_loss(
            pred, target, iou, prompt_foreground=fg, mask_contain_weight=0.0, **kwargs
        )
        self.assertNotIn("loss_mask_contain", same_stats)
        self.assertAlmostEqual(float(same), float(base), places=6)

    def test_warmup_scale(self) -> None:
        self.assertEqual(containment_warmup_scale(0, 0, 100, 2), 0.0)
        self.assertAlmostEqual(containment_warmup_scale(0, 50, 100, 2), 0.25)
        self.assertAlmostEqual(containment_warmup_scale(1, 0, 100, 2), 0.5)
        self.assertEqual(containment_warmup_scale(2, 0, 100, 2), 1.0)
        self.assertEqual(containment_warmup_scale(9, 0, 100, 2), 1.0)
        self.assertEqual(containment_warmup_scale(0, 0, 100, 0), 1.0)


class VariantDecodeTests(unittest.TestCase):
    def test_one_encoder_pass_serves_every_variant(self) -> None:
        model = _FakeSam()
        encoded = {"n": 0}
        original = model.forward_image

        def counting(images):
            encoded["n"] += 1
            return original(images)

        model.forward_image = counting
        images = torch.stack([torch.full((3, 4, 4), float(i)) for i in range(2)])
        index = torch.tensor([0, 0, 0, 1, 1, 1])
        points = torch.ones(6, 1, 2)
        labels = torch.ones(6, 1, dtype=torch.int32)
        boxes = [None, torch.tensor([0.0, 0.0, 1.0, 1.0]), None] * 2
        dense = [None, None, torch.zeros(256, 256)] * 2
        logits, iou = predict_masks_mixed(
            model, images, points, labels, boxes, dense, image_size=4, orig_hw=(4, 4), variant_index=index
        )
        self.assertEqual(encoded["n"], 1)
        self.assertEqual(tuple(logits.shape), (6, 4, 4))
        self.assertEqual(tuple(iou.shape), (6, 1))
        # image value (0 or 1) + one point + 10 for a box + 100 for a dense mask
        self.assertEqual(logits[:, 0, 0].tolist(), [1.0, 11.0, 101.0, 2.0, 12.0, 102.0])

    def test_sample_weight_scales_a_samples_loss(self) -> None:
        pred = torch.zeros(3, 8, 8)
        pred[0] = 4.0
        target = torch.zeros(3, 8, 8, dtype=torch.int64)
        iou = torch.zeros(3, 1)
        kwargs = dict(focal_weight=1.0, dice_weight=0.0, iou_weight=0.0)
        index = torch.tensor([0, 0, 1])
        base, stats = combined_loss(pred, target, iou, variant_index=index, num_samples=2, **kwargs)
        weighted, weighted_stats = combined_loss(
            pred, target, iou, variant_index=index, num_samples=2, sample_weight=torch.tensor([2.0, 1.0]), **kwargs
        )
        per_variant = stats["sample_losses"]
        first = (per_variant[0] + per_variant[1]) / 2
        second = per_variant[2]
        self.assertAlmostEqual(float(base), float((first + second) / 2), places=6)
        self.assertAlmostEqual(float(weighted), float((2.0 * first + second) / 2), places=6)
        self.assertTrue(torch.equal(weighted_stats["sample_losses"], per_variant))  # stats stay unweighted
        plain, _ = combined_loss(pred, target, iou, **kwargs)
        scaled, _ = combined_loss(pred, target, iou, sample_weight=torch.tensor([2.0, 2.0, 2.0]), **kwargs)
        self.assertAlmostEqual(float(scaled), 2.0 * float(plain), places=6)

    def test_gradient_flows_to_the_shared_image(self) -> None:
        model = _FakeSam()
        images = torch.zeros(2, 3, 4, 4, requires_grad=True)
        index = torch.tensor([0, 0, 1])
        logits, _ = predict_masks_mixed(
            model, images, torch.ones(3, 1, 2), torch.ones(3, 1, dtype=torch.int32),
            [None] * 3, [None] * 3, image_size=4, variant_index=index,
        )
        logits.sum().backward()
        self.assertIsNotNone(images.grad)


class ActivationCheckpointingTests(unittest.TestCase):
    def test_fraction_picks_blocks_spread_evenly(self) -> None:
        from sam2_segmentation_trainer.model import checkpointed_block_indices

        self.assertEqual(checkpointed_block_indices(48, 1.0), list(range(48)))
        self.assertEqual(checkpointed_block_indices(48, 0.0), [])
        half = checkpointed_block_indices(48, 0.5)
        self.assertEqual(len(half), 24)
        self.assertEqual(half[:3], [0, 2, 4])
        quarter = checkpointed_block_indices(48, 0.25)
        self.assertEqual(len(quarter), 12)
        self.assertLessEqual(max(np.diff(quarter)), 4)
        self.assertEqual(checkpointed_block_indices(48, 2.0), list(range(48)))  # clamped

    def test_only_the_chosen_blocks_are_wrapped(self) -> None:
        from sam2_segmentation_trainer.model import enable_activation_checkpointing

        class Trunk(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(8)])

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_encoder = nn.Module()
                self.image_encoder.trunk = Trunk()

        model = Model()
        self.assertEqual(enable_activation_checkpointing(model, 0.5), 4)
        wrapped = [getattr(b, "_em_activation_ckpt", False) for b in model.image_encoder.trunk.blocks]
        self.assertEqual(sum(wrapped), 4)
        self.assertEqual(wrapped[0], True)
        self.assertEqual(wrapped[1], False)
        self.assertEqual(enable_activation_checkpointing(Model(), 1.0), 8)
        self.assertEqual(enable_activation_checkpointing(Model(), 0.0), 0)


class EdgeWeightedLossTests(unittest.TestCase):
    def test_weight_curve_and_floor(self) -> None:
        d = torch.tensor([0.0, 2.5, 7.5, 100.0, float("inf")])
        w = edge_loss_weight(d, 2.5, 0.05)
        self.assertAlmostEqual(float(w[0]), 0.05)
        self.assertAlmostEqual(float(w[1]), 1 - 2.718281828 ** -1, places=4)
        self.assertGreater(float(w[2]), 0.95)
        self.assertEqual(float(w[3]), 1.0)
        self.assertEqual(float(w[4]), 1.0)
        self.assertTrue(bool((edge_loss_weight(d, 0.0, 0.05) == 1.0).all()))

    def test_unit_weights_match_the_unweighted_losses(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(2, 8, 8)
        target = (torch.rand(2, 8, 8) > 0.5).long()
        ones = torch.ones(2, 8, 8)
        self.assertTrue(torch.allclose(
            weighted_focal_loss_per_sample(pred, target, ones), weighted_focal_loss_per_sample(pred, target), atol=1e-6))
        self.assertTrue(torch.allclose(
            weighted_dice_loss(pred, target, ones), weighted_dice_loss(pred, target), atol=1e-6))

    def test_errors_in_the_edge_band_cost_less(self) -> None:
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, 4:12, 4:12] = 1
        pred = (target.float() * 20.0 - 10.0).clone()
        wrong = pred.clone()
        wrong[:, 4, 4:12] = -10.0  # the first row of the mask called background
        weight = torch.ones(1, 16, 16)
        weight[:, 3:5, :] = 0.05  # that row sits in the band around the traced edge
        full = weighted_focal_loss_per_sample(wrong, target)
        tolerant = weighted_focal_loss_per_sample(wrong, target, weight)
        self.assertLess(float(tolerant), float(full) * 0.2)
        self.assertLess(float(weighted_dice_loss(wrong, target, weight)), float(weighted_dice_loss(wrong, target)))
        # An error far from the edge still counts in full.
        far = pred.clone()
        far[:, 8, 8] = -10.0
        self.assertGreater(float(weighted_dice_loss(far, target, weight)), 0.0)

    def test_iou_target_ignores_the_band(self) -> None:
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, 4:12, 4:12] = 1
        pred = (target.float() * 20.0 - 10.0).clone()
        pred[:, 4, 4:12] = -10.0
        weight = torch.ones(1, 16, 16)
        weight[:, 3:5, :] = 0.05
        self.assertLess(float(tolerant_iou_target(pred, target, None)), 0.95)
        self.assertAlmostEqual(float(tolerant_iou_target(pred, target, weight)), 1.0, places=5)

    def test_combined_loss_averages_variants_within_a_sample_first(self) -> None:
        pred = torch.zeros(3, 8, 8)
        target = torch.zeros(3, 8, 8, dtype=torch.int64)
        iou = torch.zeros(3, 1)
        kwargs = dict(focal_weight=1.0, dice_weight=0.0, iou_weight=0.0)
        pred[0] = 4.0  # sample 0, variant a: wrong
        index = torch.tensor([0, 0, 1])
        plain, plain_stats = combined_loss(pred, target, iou, **kwargs)
        grouped, grouped_stats = combined_loss(pred, target, iou, variant_index=index, **kwargs)
        per_variant = plain_stats["sample_losses"]
        self.assertEqual(tuple(per_variant.shape), (3,))
        expected = ((per_variant[0] + per_variant[1]) / 2 + per_variant[2]) / 2
        self.assertAlmostEqual(float(grouped), float(expected), places=6)
        self.assertNotAlmostEqual(float(plain), float(grouped), places=6)
        # Passing the sample count gives the same value without reading it back from the GPU.
        counted, _ = combined_loss(pred, target, iou, variant_index=index, num_samples=2, **kwargs)
        self.assertAlmostEqual(float(counted), float(grouped), places=6)

    def test_seed_containment_is_weighted_by_trust(self) -> None:
        pred = torch.full((1, 4, 4), -2.0)
        trust = torch.zeros(1, 4, 4)
        trust[0, 0, 0] = 1.0
        trust[0, 0, 1] = 0.25
        # (1.0 * 2 + 0.25 * 2) / 1.25: the weight-normalized mean, so the core counts most
        self.assertAlmostEqual(float(mask_containment_loss_per_sample(pred, trust)[0]), 2.0)
        pred[0, 0, 0] = 5.0
        self.assertAlmostEqual(float(mask_containment_loss_per_sample(pred, trust)[0]), 0.4)


if __name__ == "__main__":
    unittest.main()
