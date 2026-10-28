"""Adjacent-section shapes: parsing ``shapes.jsonl``, the tile-pixel mapping, the neighbour pick and the dataset path."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sam2_segmentation_trainer.data.dataset import EMSegDataset
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import PromptConfig, PromptType
from sam2_segmentation_trainer.data.shapes import (
    SHAPES_FILENAME,
    VolumeShapes,
    load_volume_shapes,
    pick_neighbour_raster,
    tile_window,
)
from sam2_segmentation_trainer.data.tiles import TileContext, build_tile_context

try:
    from tests.helpers import make_crops
except ImportError:
    from helpers import make_crops


def _square(x0: float, y0: float, x1: float, y1: float) -> list:
    """A one-polygon record body: the square's centroid and its exterior ring relative to it."""
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ring = [[x0 - cx, y0 - cy], [x1 - cx, y0 - cy], [x1 - cx, y1 - cy], [x0 - cx, y1 - cy], [x0 - cx, y0 - cy]]
    return [cx, cy], [[ring]]


def _record(location_id: int, z: int, box, *, has_mask: bool = True, linked=(), hole=None) -> dict:
    centroid, polygons = _square(*box)
    if hole is not None:
        cx, cy = centroid
        hx0, hy0, hx1, hy1 = hole
        polygons[0].append([[hx0 - cx, hy0 - cy], [hx1 - cx, hy0 - cy], [hx1 - cx, hy1 - cy], [hx0 - cx, hy1 - cy]])
    return {
        "location_id": location_id, "z": z, "structure_id": 77, "structure_label": "BC", "has_mask": has_mask,
        "linked": [list(pair) for pair in linked], "centroid": centroid, "area": 1.0, "polygons": polygons,
    }


def _write(path: Path, records: list[dict]) -> Path:
    # Compact separators, as the exporter writes them (so the empty-links fast path is exercised).
    path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n", encoding="utf-8")
    return path


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class ParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_keeps_only_targets_with_a_near_neighbour_and_those_neighbours(self) -> None:
        records = [
            # target 1 on z 5: neighbours 2 (z 4), 3 (z 6, with a hole), 4 (z 8, too far) and 9 (no record)
            _record(1, 5, (0, 0, 10, 10), linked=[(2, 4), (3, 6), (4, 8), (9, 7)]),
            _record(2, 4, (0, 0, 12, 12), has_mask=False, linked=[(1, 5)]),
            _record(3, 6, (0, 0, 20, 20), has_mask=False, linked=[(1, 5)], hole=(5, 5, 15, 15)),
            _record(4, 8, (0, 0, 8, 8), has_mask=False, linked=[(1, 5)]),
            _record(5, 5, (40, 40, 50, 50)),  # a target without links
            _record(6, 3, (0, 0, 5, 5), linked=[(7, 3)]),  # a link in its own section is not adjacent
            _record(7, 3, (0, 0, 5, 5), linked=[(6, 3)]),
        ]
        shapes = load_volume_shapes(_write(self.dir / SHAPES_FILENAME, records))
        self.assertIsNotNone(shapes)
        self.assertEqual(sorted(shapes.links), [1])
        kept = [int(shapes.location_ids[row]) for row in shapes.neighbour_rows(1)]
        self.assertEqual(sorted(kept), [2, 3])
        self.assertEqual(sorted(int(i) for i in shapes.location_ids), [2, 3])  # target 1's own outline is not needed
        self.assertEqual(shapes.neighbour_rows(5), ())
        row3 = int(np.flatnonzero(shapes.location_ids == 3)[0])
        rings = shapes.rings(row3)
        self.assertEqual([exterior for _pts, exterior in rings], [True, False])
        np.testing.assert_allclose(rings[0][0].min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(rings[0][0].max(axis=0), [20.0, 20.0])
        self.assertEqual(shapes.points.dtype, np.float32)

    def test_targets_restrict_the_links(self) -> None:
        records = [
            _record(1, 5, (0, 0, 10, 10), linked=[(2, 4)]),
            _record(2, 4, (0, 0, 12, 12), linked=[(1, 5)]),
        ]
        path = _write(self.dir / SHAPES_FILENAME, records)
        both = load_volume_shapes(path)
        self.assertEqual(sorted(both.links), [1, 2])  # both have masks and each is the other's neighbour
        only_one = load_volume_shapes(path, targets={2})
        self.assertEqual(sorted(only_one.links), [2])
        self.assertEqual([int(i) for i in only_one.location_ids], [1])
        self.assertEqual(len(load_volume_shapes(path, targets=set())), 0)

    def test_missing_file_and_bad_lines(self) -> None:
        self.assertIsNone(load_volume_shapes(self.dir / "absent.jsonl"))
        path = self.dir / SHAPES_FILENAME
        good = [_record(1, 5, (0, 0, 10, 10), linked=[(2, 4)]), _record(2, 4, (0, 0, 12, 12), linked=[(1, 5)])]
        path.write_text(
            "not json\n" + "\n".join(json.dumps(r) for r in good) + '\n{"location_id": 3}\n', encoding="utf-8"
        )
        shapes = load_volume_shapes(path, targets={1})  # spaced JSON: parsed the slow way, same result
        self.assertEqual(sorted(shapes.links), [1])


class CoordinateTests(unittest.TestCase):
    def test_tile_window(self) -> None:
        self.assertEqual(tile_window("RC2_696_D1_X44544-45568_Y68608-69632"), (1, 44544, 68608))
        self.assertEqual(tile_window("RC2_17_D2_X-512-512_Y0-512"), (2, -512, 0))
        self.assertEqual(tile_window("RC2_17_D2_X-1024--512_Y-8-504"), (2, -1024, -8))
        self.assertIsNone(tile_window("no_window_here"))

    def _shapes(self, box, hole=None) -> VolumeShapes:
        with TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / SHAPES_FILENAME,
                [_record(1, 5, (0, 0, 1, 1), linked=[(2, 4)]), _record(2, 4, box, has_mask=False, linked=[(1, 5)], hole=hole)],
            )
            return load_volume_shapes(path)

    def test_a_polygon_lands_at_its_tile_pixels(self) -> None:
        shapes = self._shapes((45000, 68700, 45100, 68800), hole=(45040, 68740, 45060, 68760))
        raster = pick_neighbour_raster(shapes, "RC2_696_D1_X44544-45568_Y68608-69632", 1, (1024, 1024), placement="raw")
        x0, y0, x1, y1 = _bbox(raster)
        # volume x 45000..45100 -> tile 456..556, y 68700..68800 -> 92..192 (y down, no flip); PIL fills the edge
        self.assertTrue(455 <= x0 <= 457 and 556 <= x1 <= 558, (x0, x1))
        self.assertTrue(91 <= y0 <= 93 and 192 <= y1 <= 194, (y0, y1))
        self.assertTrue(raster[100, 470])
        self.assertFalse(raster[92 + 50, 456 + 50])  # inside the hole

    def test_negative_origin_and_downsample(self) -> None:
        # D2 window X-512-512: crop pixel = volume / 2 + 512; the window hangs over the mosaic edge.
        shapes = self._shapes((-200, 100, -100, 200))
        raster = pick_neighbour_raster(shapes, "RC2_17_D2_X-512-512_Y0-512", 1, (512, 1024), placement="raw")
        self.assertEqual(raster.shape, (512, 1024))
        x0, y0, x1, y1 = _bbox(raster)
        self.assertTrue(411 <= x0 <= 413 and 462 <= x1 <= 464, (x0, x1))
        self.assertTrue(49 <= y0 <= 51 and 100 <= y1 <= 102, (y0, y1))

    def test_an_outline_outside_the_tile_is_none(self) -> None:
        shapes = self._shapes((90000, 90000, 90100, 90100))
        self.assertIsNone(pick_neighbour_raster(shapes, "RC2_1_D1_X0-1024_Y0-1024", 1, (1024, 1024), placement="raw"))
        self.assertIsNone(pick_neighbour_raster(shapes, "RC2_1_D1_X0-1024_Y0-1024", 99, (1024, 1024)))
        with self.assertRaises(ValueError):
            pick_neighbour_raster(shapes, "RC2_1_D1_X0-1024_Y0-1024", 1, (1024, 1024), placement="volume")

    def test_centroid_placement_puts_the_neighbour_on_the_target(self) -> None:
        # Section mosaics are not registered: the neighbour (a 100 x 40 box) is written 20,000 px away. Its outline
        # is moved so its centroid sits on the target's centroid (volume 45000, 68800 -> tile 456, 192).
        with TemporaryDirectory() as tmp:
            shapes = load_volume_shapes(_write(Path(tmp) / SHAPES_FILENAME, [
                _record(1, 5, (44950, 68750, 45050, 68850), linked=[(2, 4)]),
                _record(2, 4, (65000, 70000, 65100, 70040), has_mask=False, linked=[(1, 5)]),
            ]))
        key = "RC2_696_D1_X44544-45568_Y68608-69632"
        self.assertIsNone(pick_neighbour_raster(shapes, key, 1, (1024, 1024), placement="raw"))
        raster = pick_neighbour_raster(shapes, key, 1, (1024, 1024))
        x0, y0, x1, y1 = _bbox(raster)
        self.assertTrue(405 <= x0 <= 407 and 506 <= x1 <= 508, (x0, x1))  # the neighbour's shape: 100 wide
        self.assertTrue(171 <= y0 <= 173 and 212 <= y1 <= 214, (y0, y1))  # 40 tall, centred on y 192
        # The D2 window with a negative origin: centroid (45000, 68800) / 2 - (-512, 0) = (23012, 34400) is outside.
        self.assertIsNone(pick_neighbour_raster(shapes, "RC2_17_D2_X-512-512_Y0-512", 1, (512, 1024)))


class PickTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        records = [
            _record(1, 5, (10, 10, 30, 30), linked=[(2, 4), (3, 6)]),
            _record(2, 4, (10, 10, 20, 30), has_mask=False, linked=[(1, 5)]),  # tall
            _record(3, 6, (10, 10, 30, 20), has_mask=False, linked=[(1, 5)]),  # wide
        ]
        self.shapes = load_volume_shapes(_write(Path(self._tmp.name) / SHAPES_FILENAME, records))
        self.tiles = TileContext(shapes={"V": self.shapes})
        self.key = ("V", "V_5_D1_X0-64_Y0-64")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _pick(self, *seed) -> str:
        raster = self.tiles.adjacent_shape(self.key, 1, shape=(64, 64), rng=np.random.default_rng(list(seed)))
        x0, y0, x1, y1 = _bbox(raster)
        return "tall" if (y1 - y0) > (x1 - x0) else "wide"

    def test_same_seed_epoch_index_gives_the_same_neighbour(self) -> None:
        for seed in range(20):
            self.assertEqual(self._pick(7, seed, 3, 0, 4), self._pick(7, seed, 3, 0, 4))
        picks = Counter(self._pick(7, epoch, 3, 0, 4) for epoch in range(200))
        self.assertEqual(set(picks), {"tall", "wide"})  # both neighbours are used over epochs
        self.assertGreater(min(picks.values()), 60)

    def test_without_rng_the_nearest_lower_section_is_first(self) -> None:
        raster = self.tiles.adjacent_shape(self.key, 1, shape=(64, 64))
        x0, y0, x1, y1 = _bbox(raster)
        self.assertGreater(y1 - y0, x1 - x0)  # z 4 (tall) before z 6, both |dz| = 1

    def test_no_neighbour_or_no_volume_is_none(self) -> None:
        self.assertIsNone(self.tiles.adjacent_shape(self.key, 2, shape=(64, 64)))  # 2 is not a target
        self.assertIsNone(self.tiles.adjacent_shape(("W", self.key[1]), 1, shape=(64, 64)))
        self.assertIsNone(TileContext().adjacent_shape(self.key, 1, shape=(64, 64)))


def _cfg(**probs: float) -> PromptConfig:
    names = [
        "prob_points_only", "prob_box_only", "prob_box_points", "prob_circle_seed", "prob_circle_annotator",
        "prob_circle_pull_through", "prob_adjacent_poly_seed", "prob_subset_seed", "prob_correction",
        "prob_near_correct",
    ]
    values = {name: 0.0 for name in names}
    values["variants_per_sample"] = 1
    values.update(probs)
    return PromptConfig(**values)


class DatasetTests(unittest.TestCase):
    """The synthetic export of ``tests.helpers.make_crops``: 32-pixel tiles keyed ``RC1_{i}_D{d}_X0-1_Y0-1``."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        crops = make_crops(self.root, "RC1", n=6)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        target = next(ex for ex in self.examples if ex.image_key.startswith("RC1_1_"))  # mask 8:24, z 2, D1
        self.target = target
        _write(
            crops / SHAPES_FILENAME,
            [
                _record(target.location_id, target.z, (8, 8, 24, 24), linked=[(900, 1), (901, 3)]),
                _record(900, 1, (9, 9, 23, 23), has_mask=False, linked=[(target.location_id, target.z)]),
                _record(901, 3, (10, 12, 22, 20), has_mask=False, linked=[(target.location_id, target.z)]),
            ],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _index(self, ds: EMSegDataset) -> int:
        return next(i for i, ex in enumerate(ds.examples) if ex is self.target or ex == self.target)

    def test_build_tile_context_loads_the_shapes(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tiles = build_tile_context(self.examples)
        self.assertIn("shapes RC1: 1 of", out.getvalue())
        key = (self.target.volume, self.target.image_key)
        raster = tiles.adjacent_shape(key, self.target.location_id, shape=(32, 32))
        self.assertEqual(_bbox(raster)[:2], (9, 9))  # z 1 first: lower z among |dz| = 1
        other = next(ex for ex in self.examples if ex.location_id != self.target.location_id)
        self.assertIsNone(tiles.adjacent_shape((other.volume, other.image_key), other.location_id, shape=(32, 32)))
        self.assertEqual(build_tile_context(self.examples, load_shapes=False).shapes, {})

    def test_a_missing_file_warns_and_gives_none(self) -> None:
        (self.root / "RC1" / SHAPES_FILENAME).unlink()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tiles = build_tile_context(self.examples)
        self.assertIn("WARNING: no shapes.jsonl for RC1", out.getvalue())
        key = (self.target.volume, self.target.image_key)
        self.assertIsNone(tiles.adjacent_shape(key, self.target.location_id, shape=(32, 32)))
        cfg = PromptConfig()
        ds = EMSegDataset(self.examples, split="val", image_size=32, tiles=tiles, prompt_cfg=cfg, seed=7)
        index = self._index(ds)
        for epoch in range(30):
            item = ds._mixed_item(self.target, index + 1000 * epoch)
            self.assertNotIn("adjacent_poly_seed", [v["requested_prompt_type"] for v in item["variants"]])

    def test_the_adjacent_family_is_drawn_end_to_end(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            tiles = build_tile_context(self.examples)
        ds = EMSegDataset(
            self.examples, split="val", image_size=32, tiles=tiles, prompt_cfg=_cfg(prob_adjacent_poly_seed=1.0), seed=7
        )
        item = ds[self._index(ds)]
        (variant,) = item["variants"]
        self.assertEqual(variant["requested_prompt_type"], "adjacent_poly_seed")
        self.assertEqual(variant["prompt_type"], "adjacent_poly_seed")
        seed = variant["prompt_input"].numpy() > 0
        self.assertTrue(seed.any())
        again = ds[self._index(ds)]["variants"][0]
        np.testing.assert_array_equal(again["prompt_input"].numpy(), variant["prompt_input"].numpy())

        # In the default mix the family is now drawn for this sample, and another sample still never gets it.
        mixed = EMSegDataset(self.examples, split="train", image_size=32, tiles=tiles, prompt_cfg=PromptConfig(), seed=7)
        index = self._index(mixed)
        asked = Counter()
        built = Counter()
        for trial in range(40):
            for v in mixed._mixed_item(self.target, index + 1000 * trial)["variants"]:
                asked[v["requested_prompt_type"]] += 1
                built[v["prompt_type"]] += 1
        self.assertGreater(asked["adjacent_poly_seed"], 5)
        self.assertGreater(built["adjacent_poly_seed"], 0)
        other = next(ex for ex in self.examples if ex.image_key.startswith("RC1_2_"))
        other_index = next(i for i, ex in enumerate(mixed.examples) if ex == other)
        for trial in range(20):
            kinds = [v["requested_prompt_type"] for v in mixed._mixed_item(other, other_index + 1000 * trial)["variants"]]
            self.assertNotIn("adjacent_poly_seed", kinds)

    def test_an_unfittable_shape_falls_back_to_the_circle(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            base = build_tile_context(self.examples, load_shapes=False)

        class EmptyShapeTiles(TileContext):
            def adjacent_shape(self, key, location_id, **_kwargs):
                return np.zeros((32, 32), dtype=bool)  # available (not None) but nothing to fit

        ds = EMSegDataset(
            self.examples, split="val", image_size=32, seed=7,
            tiles=EmptyShapeTiles(neighbors=base.neighbors, circles=base.circles),
            prompt_cfg=_cfg(prob_adjacent_poly_seed=1.0),
        )
        (variant,) = ds[self._index(ds)]["variants"]
        self.assertEqual(variant["requested_prompt_type"], "adjacent_poly_seed")
        self.assertEqual(variant["prompt_type"], PromptType.CIRCLE_SEED.value)  # train/prompt_fallback counts it
        self.assertIsNotNone(variant["prompt_input"])


if __name__ == "__main__":
    unittest.main()
