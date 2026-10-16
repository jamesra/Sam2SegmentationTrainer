"""Radius-relative trust of circle annotations (encoding ``edge-confidence-v6``)."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import albumentations as A
import numpy as np
import torch
from torch import nn

from annotation_crops import prompt_encoding as enc
from annotation_crops.negatives import (
    KIND_CIRCLE,
    KIND_POLYGON,
    CircleAnnotation,
    NegativeRegion,
    collect_negative_regions,
    drop_circles_over_target,
    load_circle_annotations,
    rasterize_circle,
)
from annotation_crops.prompt_encoding import (
    DEFAULT_CIRCLE_TRUST,
    CircleAnnotationTrust,
    area_resize,
    circle_negative_canvas,
    circle_trust,
    edge_confidence,
    encode_mask_input,
    encoding_differences,
    encoding_metadata,
)
from sam2_segmentation_trainer.data.dataset import EMSegDataset, augment_scene
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.data.prompts import (
    PromptConfig,
    PromptType,
    build_prompt,
    negative_region_canvases,
)
from sam2_segmentation_trainer.data.weights import leak_mask, neighbor_core_labels, region_core
from sam2_segmentation_trainer.encoding_record import (
    RECORD_NAME,
    metadata_from_settings,
    read_record,
    resolve_encoding,
    settings_from_cfg,
    sidecar_path,
    write_record,
)
from sam2_segmentation_trainer.export_serve import export_serve_checkpoint

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops

SHAPE = (96, 128)


def _rho(shape, cx, cy, radius) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) / radius


def _reference_w(rho: np.ndarray, trust: CircleAnnotationTrust) -> np.ndarray:
    ramp = 0.5 * (1.0 + np.cos(np.pi * (rho - trust.full_frac) / (trust.zero_frac - trust.full_frac)))
    return np.where(rho <= trust.full_frac, 1.0, np.where(rho >= trust.zero_frac, 0.0, ramp))


class TrustCurveTests(unittest.TestCase):
    def test_full_inside_zero_outside_and_half_at_the_midpoint(self) -> None:
        trust = CircleAnnotationTrust(0.5, 0.95)
        # A center on a pixel center makes pixel (64 + k, 48) sit at exactly rho = k / 40.
        mask, weight = circle_trust(SHAPE, 64.5, 48.5, 40.0, trust)
        row = weight[48]
        for k in (0, 5, 19, 20):  # rho <= 0.5
            self.assertEqual(float(row[64 + k]), 1.0, k)
        self.assertAlmostEqual(float(row[64 + 29]), 0.5, places=6)  # rho 0.725, midway between 0.5 and 0.95
        self.assertEqual(float(row[64 + 38]), 0.0)  # rho 0.95: still claimed, no trust left
        self.assertTrue(bool(mask[48, 64 + 38]))
        for k in (39, 40, 55):
            self.assertEqual(float(row[64 + k]), 0.0, k)
            self.assertFalse(bool(mask[48, 64 + k]), k)
        self.assertTrue(np.all(np.diff(row[64 : 64 + 40]) <= 0))  # monotone falling

    def test_the_analytic_map_matches_the_formula_everywhere(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(25):
            cx, cy = rng.uniform(-30, 160), rng.uniform(-30, 130)
            radius = rng.uniform(3, 200)
            full = rng.uniform(0, 0.8)
            trust = CircleAnnotationTrust(full, rng.uniform(full + 0.05, 1.3))
            mask, weight = circle_trust(SHAPE, cx, cy, radius, trust)
            rho = _rho(SHAPE, cx, cy, radius)
            expected = _reference_w(rho, trust)
            claimed = rho <= trust.zero_frac
            # Float32 rounding may flip pixels sitting exactly on a threshold.
            self.assertLessEqual(int((mask != claimed).sum()), 2)
            self.assertLess(float(np.abs(weight - expected * claimed).max()), 1e-3)
            self.assertEqual(float(weight[~mask].max(initial=0.0)), 0.0)

    def test_canvas_is_minus_seed_logit_times_w(self) -> None:
        claim = circle_negative_canvas(SHAPE, 60.0, 40.0, 30.0, CircleAnnotationTrust(0.5, 0.95), 10.0)
        _mask, weight = circle_trust(SHAPE, 60.0, 40.0, 30.0, CircleAnnotationTrust(0.5, 0.95))
        np.testing.assert_allclose(claim.canvas, -10.0 * weight)
        self.assertEqual(float(claim.canvas.min()), -10.0)
        self.assertEqual(claim.canvas.dtype, np.float32)

    def test_a_circle_off_the_tile_claims_nothing(self) -> None:
        mask, weight = circle_trust(SHAPE, 500.0, 500.0, 20.0, DEFAULT_CIRCLE_TRUST)
        self.assertFalse(mask.any())
        self.assertFalse(weight.any())
        mask, _ = circle_trust(SHAPE, 10.0, 10.0, 0.0, DEFAULT_CIRCLE_TRUST)
        self.assertFalse(mask.any())

    def test_claimed_region_at_the_defaults_is_the_old_095_scaled_raster(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            cx, cy, radius = rng.uniform(-10, 140), rng.uniform(-10, 110), rng.uniform(2, 80)
            old = rasterize_circle(SHAPE, cx, cy, radius * 0.95)  # v5: the loader shrank the true radius by 0.95
            mask, _ = circle_trust(SHAPE, cx, cy, radius, DEFAULT_CIRCLE_TRUST)
            self.assertLessEqual(int((old != mask).sum()), 2)  # edge pixels only (identical in practice)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_values_reach_the_encoder(self) -> None:
        self.assertEqual(PromptConfig().circle_annotation_trust, CircleAnnotationTrust(0.5, 0.95))
        cfg = PromptConfig.from_cfg({"prompt": {"circle_annotation_trust": {"full_frac": 0.3, "zero_frac": 0.8}}})
        self.assertEqual(cfg.circle_annotation_trust, CircleAnnotationTrust(0.3, 0.8))
        circle = CircleAnnotation(7, 64.0, 48.0, 40.0, source="annotated")
        with TemporaryDirectory() as tmp:
            regions = collect_negative_regions("k", 1, [], [circle], Path(tmp), SHAPE, trust=cfg.circle_annotation_trust)
        (region,) = regions
        np.testing.assert_array_equal(region.mask, _rho(SHAPE, 64.0, 48.0, 40.0) <= 0.8)
        canvas = negative_region_canvases(regions, cfg)[0].canvas
        self.assertEqual(float(canvas[48, 64]), -cfg.seed_logit)
        self.assertTrue(-cfg.seed_logit < float(canvas[48, 64 + 30]) < 0.0)  # rho 0.75 > 0.3: tapered, not full
        dense = build_prompt(
            PromptType.POINTS,
            target_mask=np.pad(np.ones((4, 4), bool), ((2, SHAPE[0] - 6), (100, SHAPE[1] - 104))),
            circle_mask=None, regions=regions, cfg=cfg, rng=np.random.default_rng(0),
        ).dense_logits
        self.assertIsNotNone(dense)
        self.assertLess(float(dense.min()), -cfg.seed_logit * 0.9)

    def test_the_yaml_ships_the_defaults(self) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "sam2_segmentation_trainer/config/em_finetune.yaml")
        self.assertEqual(
            PromptConfig.from_cfg(cfg).circle_annotation_trust, DEFAULT_CIRCLE_TRUST
        )
        self.assertEqual(settings_from_cfg(cfg)["circle_annotation_trust"], DEFAULT_CIRCLE_TRUST)

    def test_config_rejects_full_not_below_zero(self) -> None:
        for full, zero in ((0.9, 0.5), (0.5, 0.5), (-0.1, 0.5), (0.2, float("inf"))):
            with self.assertRaises(ValueError, msg=str((full, zero))):
                PromptConfig.from_cfg({"prompt": {"circle_annotation_trust": {"full_frac": full, "zero_frac": zero}}})
            with self.assertRaises(ValueError):
                CircleAnnotationTrust(full, zero)
        CircleAnnotationTrust(0.0, 0.01)  # the boundary full_frac == 0 is allowed


class LoadingTests(unittest.TestCase):
    def _load(self, file_scale: float, scale: float | None):
        entries = [
            {"location_id": 1, "center_x": 10.0, "center_y": 10.0, "radius": 20.0, "source": "annotated"},
            {"location_id": 2, "center_x": 50.0, "center_y": 10.0, "radius": 20.0, "source": "inscribed"},
        ]
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "circle_annotations.json").write_text(
                json.dumps({"version": 1, "diameter_scale": file_scale, "grid_cells": {"k": entries}}), encoding="utf-8"
            )
            return {c.source: c for c in load_circle_annotations(Path(tmp), scale)["k"]}

    def test_annotated_radii_are_not_shrunk_while_inscribed_radii_are(self) -> None:
        circles = self._load(1.0, 0.95)
        self.assertEqual(circles["annotated"].radius, 20.0)
        self.assertAlmostEqual(circles["inscribed"].radius, 19.0)

    def test_an_old_file_is_divided_by_its_scale_for_annotated_only(self) -> None:
        circles = self._load(0.8, 0.95)
        self.assertAlmostEqual(circles["annotated"].radius, 25.0)
        self.assertAlmostEqual(circles["inscribed"].radius, 20.0 * 0.95 / 0.8)

    def test_a_seed_circle_from_an_annotated_entry_is_unchanged(self) -> None:
        annotated = self._load(1.0, 0.95)["annotated"]
        self.assertAlmostEqual(annotated.seed_radius(0.95), 19.0)  # what v5 drew
        self.assertAlmostEqual(self._load(1.0, 0.95)["inscribed"].seed_radius(0.95), 19.0)

    def test_negatives_import_needs_no_scipy(self) -> None:
        code = "import annotation_crops.negatives, sys; assert 'scipy' not in sys.modules"
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.circle = CircleAnnotation(7, 64.0, 48.0, 40.0, source="annotated")
        with TemporaryDirectory() as tmp:
            self.regions = collect_negative_regions("k", 1, [], [self.circle], Path(tmp), SHAPE)

    def test_core_is_rho_within_full_frac(self) -> None:
        (region,) = self.regions
        expected = _rho(SHAPE, 64.0, 48.0, 40.0) <= DEFAULT_CIRCLE_TRUST.full_frac
        self.assertLessEqual(int((region_core(region, 10.0) != expected).sum()), 2)
        self.assertTrue(region_core(region, 10.0)[48, 64])

    def test_leak_and_core_labels_use_the_circle_core_and_the_polygon_band(self) -> None:
        polygon = np.zeros(SHAPE, bool)
        polygon[5:40, 5:40] = True
        target = np.zeros(SHAPE, bool)
        target[80:90, 100:110] = True
        regions = [NegativeRegion(1, KIND_POLYGON, polygon), self.regions[0]]
        leak = leak_mask(regions, target, 10.0)
        labels = neighbor_core_labels(regions, target, 10.0)
        circle_core = region_core(self.regions[0], 10.0)
        polygon_core = np.zeros(SHAPE, bool)
        polygon_core[15:30, 15:30] = True  # 10 px band, measured to the edge between pixels
        self.assertTrue(np.all(labels[circle_core & ~target] == 2))
        self.assertTrue(np.all(leak[circle_core & ~target]))
        self.assertEqual(int(labels[20, 20]), 1)  # the polygon's core, 10 px inside its edge
        self.assertEqual(int(labels[7, 7]), 0)  # inside the polygon but within the band
        # The circle's claimed region is wider than its core, and only the core counts.
        self.assertGreater(int(self.regions[0].mask.sum()), int(circle_core.sum()))
        self.assertFalse(leak[48, 64 + 30])  # rho 0.75: claimed, not core

    def test_bare_masks_are_still_polygons(self) -> None:
        polygon = np.zeros(SHAPE, bool)
        polygon[5:40, 5:40] = True
        self.assertTrue(np.array_equal(region_core(polygon, 10.0), region_core(NegativeRegion(1, KIND_POLYGON, polygon), 10.0)))

    def test_background_points_come_from_the_core(self) -> None:
        cfg = PromptConfig(bg_point_prob_circle=1.0, bg_point_prob_polygon=0.0, context_dropout=0.0)
        target = np.zeros(SHAPE, bool)
        target[80:90, 100:110] = True
        core = region_core(self.regions[0], 10.0)
        for seed in range(20):
            sample = build_prompt(
                PromptType.POINTS, target_mask=target, circle_mask=None, regions=self.regions, cfg=cfg,
                rng=np.random.default_rng(seed),
            )
            background = sample.point_coords[sample.point_labels == 0]
            self.assertGreater(len(background), 0)
            for x, y in background:
                self.assertTrue(core[int(y), int(x)], (x, y))


class OverlapRuleTests(unittest.TestCase):
    def test_overlap_is_decided_by_the_claimed_region(self) -> None:
        circle = CircleAnnotation(7, 40.0, 40.0, 30.0, source="annotated")
        target = np.zeros((96, 96), bool)
        target[40, 65:68] = True  # rho 0.83 .. 0.9: inside the default 0.95 claim, outside a 0.8 claim
        with TemporaryDirectory() as tmp:
            default = collect_negative_regions("k", 1, [], [circle], Path(tmp), (96, 96))
            narrow = collect_negative_regions(
                "k", 1, [], [circle], Path(tmp), (96, 96), trust=CircleAnnotationTrust(0.2, 0.8)
            )
        self.assertEqual(len(drop_circles_over_target(default, target)), 0)  # dropped: it touches the target
        self.assertEqual(len(drop_circles_over_target(narrow, target)), 1)  # kept: its claim stops short of it


class SeedPromptTests(unittest.TestCase):
    def test_a_circle_seed_prompt_is_the_v5_encoding(self) -> None:
        circle = rasterize_circle(SHAPE, 64.0, 48.0, 30.0)
        target = np.zeros(SHAPE, bool)
        target[30:70, 40:90] = True
        cfg = PromptConfig(context_dropout=0.0)
        sample = build_prompt(
            PromptType.CIRCLE_SEED, target_mask=target, circle_mask=circle, regions=[], cfg=cfg,
            rng=np.random.default_rng(3),
        )
        v5 = area_resize((cfg.seed_logit * edge_confidence(circle, cfg.edge_tau_px, cfg.edge_tau_depth_frac)), enc.LOW_RES)
        self.assertEqual(sample.dense_logits.tobytes(), v5.tobytes())

    def test_a_polygon_negative_keeps_the_tau_curve(self) -> None:
        polygon = np.zeros(SHAPE, bool)
        polygon[10:60, 10:60] = True
        region = NegativeRegion(1, KIND_POLYGON, polygon)
        cfg = PromptConfig()
        (canvas,) = negative_region_canvases([region], cfg)
        np.testing.assert_array_equal(canvas.canvas, -np.float32(cfg.seed_logit) * edge_confidence(polygon, cfg.edge_tau_px, cfg.edge_tau_depth_frac))

    def test_the_circle_negative_is_encoded_with_its_trust_not_the_tau_curve(self) -> None:
        circle = CircleAnnotation(7, 64.0, 48.0, 30.0, source="annotated")
        with TemporaryDirectory() as tmp:
            regions = collect_negative_regions("k", 1, [], [circle], Path(tmp), SHAPE)
        claim = circle_negative_canvas(SHAPE, 64.0, 48.0, 30.0, DEFAULT_CIRCLE_TRUST, 10.0)
        dense = encode_mask_input(SHAPE, negative_canvases=negative_region_canvases(regions, PromptConfig()))
        np.testing.assert_array_equal(dense, area_resize(claim.canvas, enc.LOW_RES))


class AugmentationTests(unittest.TestCase):
    def test_the_trust_map_stays_registered_through_flips_rotations_and_elastic_warps(self) -> None:
        circle = CircleAnnotation(7, 40.0, 30.0, 25.0, source="annotated")
        with TemporaryDirectory() as tmp:
            regions = collect_negative_regions("k", 1, [], [circle], Path(tmp), SHAPE)
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        transform = A.Compose(
            [A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.ElasticTransform(alpha=30, sigma=5, p=1.0)]
        )
        target = np.zeros(SHAPE, bool)
        target[0:3, 0:3] = True
        for _ in range(10):
            _image, _target, regions_aug, _extras = augment_scene(transform, rgb, target, regions)
            (aug,) = regions_aug
            self.assertEqual(aug.weight.dtype, np.float32)
            self.assertTrue(np.all(aug.weight[~aug.mask] == 0))
            self.assertTrue(np.all(aug.core <= aug.mask))
            self.assertGreater(int(aug.core.sum()), 0)
            self.assertTrue(set(np.unique(aug.weight)) - {0.0, 1.0})  # the taper survived (nearest warp keeps values)
            self.assertAlmostEqual(float(aug.weight.max()), 1.0)

    def test_regions_that_leave_the_image_are_dropped_with_their_weights(self) -> None:
        a = CircleAnnotation(7, 40.0, 30.0, 20.0, source="annotated")
        b = CircleAnnotation(8, 500.0, 500.0, 20.0, source="annotated")
        c = CircleAnnotation(9, 90.0, 60.0, 15.0, source="annotated")
        with TemporaryDirectory() as tmp:
            regions = collect_negative_regions("k", 1, [], [a, b, c], Path(tmp), SHAPE)
        self.assertEqual([r.location_id for r in regions], [7, 9])
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        _i, _t, regions_aug, _e = augment_scene(A.Compose([A.HorizontalFlip(p=1.0)]), rgb, np.ones(SHAPE, bool), regions)
        self.assertEqual([r.location_id for r in regions_aug], [7, 9])
        for before, after in zip(regions, regions_aug):
            np.testing.assert_array_equal(after.weight, before.weight[:, ::-1])


class DatasetTests(unittest.TestCase):
    """An annotated circle in the tile reaches the prompt and the cores through the whole dataset path."""

    def _dataset(self, root: Path, trust: CircleAnnotationTrust, **kwargs) -> EMSegDataset:
        crops = make_crops(root, "RC1", n=3)
        examples = index_volumes(volumes=["RC1"], root=root)
        first = examples[0]
        entry = {
            "location_id": 99999, "center_x": 4.0, "center_y": 4.0, "radius": 4.0,
            "structure_label": "", "has_polygon_mask": False, "source": "annotated",
        }
        (crops / "circle_annotations.json").write_text(
            json.dumps({"version": 1, "diameter_scale": 1.0, "grid_cells": {first.image_key: [entry]}}), encoding="utf-8"
        )
        context = build_tile_context(examples, circle_scale=0.95)
        cfg = PromptConfig(variants_per_sample=1, circle_annotation_trust=trust)
        self.first = first
        return EMSegDataset([first], split="val", image_size=32, tiles=context, prompt_cfg=cfg, seed=1,
                            val_prompt="circle_mask", **kwargs)

    def test_the_trust_fractions_change_the_dense_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            default = self._dataset(Path(tmp) / "a", DEFAULT_CIRCLE_TRUST)[0]["variants"][0]["dense_mask"]
        with TemporaryDirectory() as tmp:
            narrow = self._dataset(Path(tmp) / "b", CircleAnnotationTrust(0.1, 0.4))[0]["variants"][0]["dense_mask"]
        self.assertFalse(np.array_equal(default.numpy(), narrow.numpy()))
        self.assertLess(float(default.min()), -9.0)  # the full-trust core of the circle
        # The circle (center 4,4 radius 4, so claimed out to 3.8 px) is a negative in the top-left corner only.
        self.assertEqual(float(default[-1, -1]), 0.0)

    def test_the_neighbor_core_is_the_full_trust_disc_and_the_seed_circle_is_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp), DEFAULT_CIRCLE_TRUST, core_band_px=10.0)
            item = dataset[0]
            expected = _rho((32, 32), 4.0, 4.0, 4.0) <= DEFAULT_CIRCLE_TRUST.full_frac
            core = item["neighbor_core"].numpy() > 0
            self.assertLessEqual(int((core != expected).sum()), 2)
            self.assertTrue(core.any())
            # The target's own circle is the inscribed seed circle: scaled by circle_diameter_scale, as in v5.
            seed = item["variants"][0]["prompt_input"].numpy().astype(bool)
            self.assertTrue(seed.any())
            self.assertLess(int(seed.sum()), int(np.ones((16, 16)).sum()))


class RecordRoundTripTests(unittest.TestCase):
    CONFIG = (
        "prompt:\n  circle_diameter_scale: 0.9\n  circle_annotation_trust:\n    full_frac: 0.4\n    zero_frac: 0.9\n"
    )

    def test_metadata_carries_the_trust_and_the_v6_version(self) -> None:
        meta = encoding_metadata(0.95, circle_annotation_trust=CircleAnnotationTrust(0.4, 0.9))
        self.assertEqual(meta["prompt_encoding"], "edge-confidence-v6")
        self.assertEqual(meta["circle_annotation_trust"], {"full_frac": 0.4, "zero_frac": 0.9, "ramp": "cosine"})
        self.assertEqual(encoding_metadata(0.95)["circle_annotation_trust"]["full_frac"], 0.5)

    def test_differences_cover_the_trust(self) -> None:
        kwargs = dict(
            edge_tau_px=enc.EDGE_TAU_PX, edge_tau_depth_frac=enc.EDGE_TAU_DEPTH_FRAC, seed_logit=enc.SEED_LOGIT,
            correction_logit=enc.CORRECTION_LOGIT, circle_diameter_scale=0.95, default_circle_diameter_scale=0.95,
        )
        self.assertEqual(encoding_differences(**kwargs), [])
        messages = encoding_differences(circle_annotation_trust=CircleAnnotationTrust(0.4, 0.9), **kwargs)
        self.assertEqual(len(messages), 2)
        self.assertIn("full_frac is 0.4", messages[0])

    def test_train_to_export_to_eval(self) -> None:
        from omegaconf import OmegaConf

        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            cfg = OmegaConf.create(self.CONFIG)
            OmegaConf.save(cfg, run / "config.yaml")
            # Training start: the run's own record, so a raw best_model.pt has one.
            write_record(run / RECORD_NAME, metadata_from_settings(settings_from_cfg(cfg)))
            record, source = read_record(run / "best_model.pt")
            self.assertEqual(record["circle_annotation_trust"], {"full_frac": 0.4, "zero_frac": 0.9, "ramp": "cosine"})
            self.assertEqual(Path(source).name, RECORD_NAME)
            # Export: the .pt embeds the dict and a sidecar holds the same one.
            out = Path(tmp) / "serve" / "best_TEM_model.pt"
            with mock.patch("sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)), \
                    contextlib.redirect_stdout(io.StringIO()):
                export_serve_checkpoint(
                    finetuned=run / "best_model.pt", base_ckpt=Path("b.pt"), out=out, model_id="v1", epoch=0
                )
            embedded = torch.load(out, weights_only=True)["prompt_encoding"]
            sidecar = sidecar_path(out)
            self.assertEqual(sidecar.name, "best_TEM_model.prompt_encoding.json")
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), embedded)
            self.assertEqual(embedded, record)
            # Eval reads the sidecar; without it the embedded copy; without either, the run's neighbour file.
            log: list[str] = []
            sidecar_record, sidecar_source = read_record(out)
            self.assertEqual(sidecar_source, str(sidecar))
            resolved = resolve_encoding(sidecar_record, source=sidecar_source, log=log.append)
            self.assertEqual(resolved.circle_annotation_trust, CircleAnnotationTrust(0.4, 0.9))
            self.assertEqual(resolved.circle_diameter_scale, 0.9)
            sidecar.unlink()
            embedded_record, embedded_source = read_record(out)
            self.assertEqual(embedded_record, embedded)
            self.assertIn("embedded", embedded_source)

    def test_command_line_overrides_win_and_are_logged(self) -> None:
        record = encoding_metadata(0.9, circle_annotation_trust=CircleAnnotationTrust(0.4, 0.9))
        log: list[str] = []
        resolved = resolve_encoding(record, source="m.json", full_frac=0.3, circle_scale=0.8, log=log.append)
        self.assertEqual(resolved.circle_annotation_trust, CircleAnnotationTrust(0.3, 0.9))
        self.assertEqual(resolved.circle_diameter_scale, 0.8)
        text = "\n".join(log)
        self.assertIn("override: circle_annotation_trust (0.4, 0.9) -> (0.3, 0.9)", text)
        self.assertIn("override: circle_diameter_scale 0.9 -> 0.8", text)
        self.assertIn("prompt.circle_annotation_trust.full_frac=0.3", resolved.cfg_overrides())
        with self.assertRaises(ValueError):
            resolve_encoding(record, full_frac=0.95, log=lambda _m: None)

    def test_no_record_means_no_config_overrides(self) -> None:
        resolved = resolve_encoding(None, log=lambda _m: None)
        self.assertEqual(resolved.cfg_overrides(), [])
        self.assertEqual(resolved.circle_annotation_trust, DEFAULT_CIRCLE_TRUST)

    def test_a_record_of_another_version_is_warned_about(self) -> None:
        log: list[str] = []
        resolve_encoding({"prompt_encoding": "edge-confidence-v5"}, log=log.append)
        self.assertTrue(any("WARNING" in line and "v5" in line for line in log))


if __name__ == "__main__":
    unittest.main()
