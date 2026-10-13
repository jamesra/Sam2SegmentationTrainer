from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from annotation_crops.negatives import CIRCLE_FILE_NAME
from sam2_segmentation_trainer.data.dataset import EMSegDataset, EpochCounter, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import MASK_PROMPT_LOGIT, PromptConfig
from sam2_segmentation_trainer.data.tiles import build_tile_context, circle_target_examples
from sam2_segmentation_trainer.losses import containment_warmup_scale, mask_containment_loss_per_sample
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


def _cfg(**kwargs: float) -> PromptConfig:
    values = {
        "prob_points_only": 0.0,
        "prob_box_only": 0.0,
        "prob_box_points": 0.0,
        "prob_circle_mask": 0.0,
        "prob_full_mask": 0.0,
        "prob_negative_only": 0.0,
    }
    values.update(kwargs)
    return PromptConfig(**values)


def _write_circles(root: Path, volume: str, cells: dict) -> None:
    path = root / volume / "AnnotationCrops" / CIRCLE_FILE_NAME
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

    def _dataset(self, cfg: PromptConfig, examples=None, epoch: EpochCounter | None = None) -> EMSegDataset:
        examples = self.examples if examples is None else examples
        return EMSegDataset(
            examples,
            split="val",
            image_size=32,
            tiles=build_tile_context(self.examples),
            prompt_cfg=cfg,
            seed=7,
            epoch=epoch,
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
        self.assertEqual(item["prompt_type"], "centroid")
        self.assertIsNone(item["boxes"])
        self.assertIsNone(item["dense_mask"])
        batch = collate_em([ds[0], ds[1]])
        self.assertEqual(tuple(batch["point"].shape), (2, 1, 2))
        self.assertEqual(tuple(batch["prompt_fg"].shape), (2, 32, 32))
        self.assertEqual(batch["prompt_type"], ["centroid", "centroid"])

    def test_neighbor_becomes_negative_in_a_full_mask_prompt(self) -> None:
        ds = self._dataset(_cfg(prob_full_mask=1.0))
        item = self._item(ds, self.shared[0].location_id)
        self.assertEqual(item["prompt_type"], "full_mask")
        dense = item["dense_mask"]
        self.assertEqual(tuple(dense.shape), (256, 256))
        self.assertEqual(set(dense.unique().tolist()), {-MASK_PROMPT_LOGIT, 0.0, MASK_PROMPT_LOGIT})
        self.assertEqual(float(dense[24, 24]), -MASK_PROMPT_LOGIT)  # neighbor tile pixels 2:6 -> low-res rows 16:48
        self.assertEqual(float(dense[128, 128]), MASK_PROMPT_LOGIT)  # target interior
        self.assertEqual(int(item["prompt_fg"].sum()), 16 * 16)
        self.assertIsNone(item["boxes"])

    def test_tile_without_neighbors_uses_no_mask_embedding(self) -> None:
        ds = self._dataset(_cfg(prob_points_only=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        self.assertEqual(item["prompt_type"], "points")
        self.assertIsNone(item["dense_mask"])
        self.assertTrue((item["point_label"] == 1).all())

    def test_points_prompt_adds_background_points_in_the_neighbor(self) -> None:
        ds = self._dataset(_cfg(prob_points_only=1.0, bg_point_prob_polygon=1.0))
        item = self._item(ds, self.shared[0].location_id)
        labels = item["point_label"]
        self.assertTrue((labels == 0).any() and (labels == 1).any())
        for (x, y), label in zip(item["point"].tolist(), labels.tolist()):
            if label == 0:
                self.assertTrue(2 <= x < 6 and 2 <= y < 6)

    def test_negative_only_has_only_background_points(self) -> None:
        ds = self._dataset(_cfg(prob_negative_only=1.0))
        item = self._item(ds, self.shared[0].location_id)
        self.assertEqual(item["prompt_type"], "negative_only")
        self.assertTrue((item["point_label"] == 0).all())
        self.assertIsNone(item["boxes"])
        self.assertIsNotNone(item["dense_mask"])

    def test_negative_only_falls_back_when_the_tile_is_alone(self) -> None:
        ds = self._dataset(_cfg(prob_negative_only=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        self.assertEqual(item["prompt_type"], "full_mask")

    def test_box_prompt_contains_the_object(self) -> None:
        ds = self._dataset(_cfg(prob_box_only=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        box = item["boxes"].tolist()
        self.assertTrue(box[0] <= 8 and box[1] <= 8 and box[2] >= 24 and box[3] >= 24)
        self.assertEqual(item["point"].shape[0], 0)

    def test_circle_mask_prompt_uses_the_annotation_circle(self) -> None:
        loc = self.alone[0].location_id
        _write_circles(
            self.root,
            "RC1",
            {self.alone_key: [{"location_id": loc, "center_x": 16.0, "center_y": 16.0, "radius": 4.0,
                               "has_polygon_mask": True, "source": "inscribed"}]},
        )
        ds = self._dataset(_cfg(prob_circle_mask=1.0))
        item = self._item(ds, loc, self.alone_key)
        self.assertEqual(item["prompt_type"], "circle_mask")
        self.assertEqual(item["point"].shape, (1, 2))
        self.assertAlmostEqual(float(item["point"][0, 0]), 15.5, places=1)
        self.assertGreater(int(item["prompt_fg"].sum()), 20)
        self.assertLess(int(item["prompt_fg"].sum()), int(item["mask"].sum()))
        self.assertEqual(float(item["dense_mask"][128, 128]), MASK_PROMPT_LOGIT)

    def test_circle_mask_falls_back_without_a_circle(self) -> None:
        ds = self._dataset(_cfg(prob_circle_mask=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        self.assertEqual(item["prompt_type"], "full_mask")

    def test_unmasked_circle_in_the_tile_is_a_negative(self) -> None:
        _write_circles(
            self.root,
            "RC1",
            {self.alone_key: [{"location_id": 9999, "center_x": 4.0, "center_y": 28.0, "radius": 3.0}]},
        )
        ds = self._dataset(_cfg(prob_full_mask=1.0))
        item = self._item(ds, self.alone[0].location_id, self.alone_key)
        self.assertEqual(float(item["dense_mask"][int(28 / 32 * 256), int(4 / 32 * 256)]), -MASK_PROMPT_LOGIT)

    def test_prompts_replay_for_the_same_epoch_index_and_seed(self) -> None:
        counter = EpochCounter()
        ds = self._dataset(_cfg(prob_points_only=1.0, bg_point_prob_polygon=1.0), epoch=counter)
        counter.set(3)
        first = self._item(ds, self.shared[0].location_id)
        second = self._item(ds, self.shared[0].location_id)
        self.assertTrue(torch.equal(first["point"], second["point"]))
        self.assertEqual(counter.get(), 3)

    def test_circle_only_targets(self) -> None:
        _write_circles(
            self.root,
            "RC1",
            {
                self.key: [
                    {"location_id": 9001, "center_x": 28.0, "center_y": 28.0, "radius": 4.5,
                     "has_polygon_mask": False, "source": "annotated"},
                    {"location_id": 9002, "center_x": 28.0, "center_y": 20.0, "radius": 1.0,
                     "has_polygon_mask": False, "source": "annotated"},
                    {"location_id": self.shared[0].location_id, "center_x": 16.0, "center_y": 16.0,
                     "radius": 5.0, "has_polygon_mask": True, "source": "inscribed"},
                ]
            },
        )
        context = build_tile_context(self.examples)
        extra = circle_target_examples(self.examples, context, max_fraction=1.0, seed=1)
        self.assertEqual([ex.location_id for ex in extra], [9001])  # 9002 is too small, the third has a mask
        self.assertEqual(extra[0].target_circle, (28.0, 28.0, 4.5))
        self.assertEqual(extra[0].split_key, "RC1:9001")
        self.assertEqual(
            circle_target_examples(self.examples, context, max_fraction=0.0, seed=1), []
        )

        ds = EMSegDataset(
            extra,
            split="val",
            image_size=32,
            tiles=context,
            prompt_cfg=_cfg(prob_full_mask=1.0),
            seed=1,
        )
        item = ds[0]
        self.assertEqual(item["prompt_type"], "negative_only")
        self.assertTrue((item["point_label"] == 0).all())
        self.assertGreater(int(item["mask"].sum()), 20)
        self.assertEqual(float(item["mask"][28, 28]), 1.0)

    def test_collate_pads_points_and_keeps_lists(self) -> None:
        def item(points: int, box: bool, dense: bool) -> dict:
            from sam2_segmentation_trainer.data.meta import SampleMeta

            return {
                "image": torch.zeros(3, 4, 4),
                "mask": torch.zeros(4, 4, dtype=torch.int64),
                "point": torch.arange(points * 2, dtype=torch.float32).reshape(points, 2),
                "point_label": torch.ones(points, dtype=torch.int32),
                "boxes": torch.zeros(4) if box else None,
                "dense_mask": torch.zeros(256, 256) if dense else None,
                "prompt_fg": torch.ones(4, 4, dtype=torch.uint8) if dense else None,
                "prompt_type": "x",
                "orig_hw": (4, 4),
                "sample_meta": SampleMeta("V", "K", 1, 1, "c", "V:1"),
            }

        batch = collate_em([item(1, False, False), item(3, True, False), item(0, True, True)])
        self.assertEqual(tuple(batch["point"].shape), (3, 3, 2))
        self.assertEqual(batch["point_label"].tolist(), [[1, -1, -1], [1, 1, 1], [-1, -1, -1]])
        self.assertEqual([b is not None for b in batch["boxes"]], [False, True, True])
        self.assertEqual([d is not None for d in batch["dense_mask"]], [False, False, True])
        self.assertEqual(batch["prompt_fg"].sum(dim=(1, 2)).tolist(), [0, 0, 16])


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


if __name__ == "__main__":
    unittest.main()
