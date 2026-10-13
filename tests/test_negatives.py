from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from annotation_crops.negatives import (
    CIRCLE_FILE_NAME,
    KIND_CIRCLE,
    KIND_POLYGON,
    CircleAnnotation,
    MaskCache,
    circle_file_stamp,
    collect_negative_regions,
    combine_signed,
    composite_negative_mask,
    load_circle_annotations,
    rasterize_circle,
    stamp_negative_regions,
)

try:
    from tests.helpers import write_png
except ImportError:
    from helpers import write_png

SHAPE = (32, 32)


def _write_mask(mask_dir: Path, key: str, location_id: int, rows: slice, cols: slice) -> None:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[rows, cols] = 255
    write_png(mask_dir / f"{key}_{location_id}.png", mask)


class RasterizeCircleTests(unittest.TestCase):
    def test_disc_is_centered_on_pixel_centers(self) -> None:
        disc = rasterize_circle((10, 10), 5.0, 5.0, 2.0)
        self.assertTrue(disc[4, 4] and disc[5, 5] and disc[4, 5] and disc[5, 4])
        self.assertFalse(disc[0, 0])
        self.assertEqual(disc.sum(), int(disc.T.sum()))

    def test_disc_outside_image_is_empty_and_partial_is_clipped(self) -> None:
        self.assertFalse(rasterize_circle((10, 10), 50.0, 50.0, 3.0).any())
        partial = rasterize_circle((10, 10), 0.0, 0.0, 4.0)
        self.assertTrue(partial[0, 0])
        self.assertFalse(partial[9, 9])

    def test_non_positive_radius_is_empty(self) -> None:
        self.assertFalse(rasterize_circle((8, 8), 4.0, 4.0, 0.0).any())


class LoadCircleAnnotationsTests(unittest.TestCase):
    def test_missing_and_bad_files_give_no_circles(self) -> None:
        with TemporaryDirectory() as tmp:
            crops = Path(tmp)
            self.assertEqual(load_circle_annotations(crops), {})
            self.assertIsNone(circle_file_stamp(crops))
            (crops / CIRCLE_FILE_NAME).write_text("not json", encoding="utf-8")
            self.assertEqual(load_circle_annotations(crops), {})
            (crops / CIRCLE_FILE_NAME).write_text(json.dumps({"version": 2, "grid_cells": {}}))
            self.assertEqual(load_circle_annotations(crops), {})

    def test_reads_entries_and_skips_malformed(self) -> None:
        payload = {
            "version": 1,
            "diameter_scale": 0.8,
            "grid_cells": {
                "K": [
                    {
                        "location_id": 7,
                        "center_x": 10.5,
                        "center_y": 12.0,
                        "radius": 4.0,
                        "structure_label": "cell",
                        "has_polygon_mask": True,
                        "source": "inscribed",
                    },
                    {"location_id": 8},
                ],
                "EMPTY": [],
            },
        }
        with TemporaryDirectory() as tmp:
            crops = Path(tmp)
            (crops / CIRCLE_FILE_NAME).write_text(json.dumps(payload), encoding="utf-8")
            index = load_circle_annotations(crops)
            self.assertEqual(list(index), ["K"])
            circle = index["K"][0]
            self.assertEqual(
                circle,
                CircleAnnotation(7, 10.5, 12.0, 4.0, "cell", True, "inscribed"),
            )
            self.assertEqual(len(index["K"]), 1)
            self.assertIsNotNone(circle_file_stamp(crops))

    def test_center_in(self) -> None:
        circle = CircleAnnotation(1, 40.0, 5.0, 3.0)
        self.assertFalse(circle.center_in((32, 32)))
        self.assertTrue(CircleAnnotation(1, 31.9, 0.0, 3.0).center_in((32, 32)))


class NegativeCompositeTests(unittest.TestCase):
    def test_neighbor_masks_and_unmasked_circles_become_regions(self) -> None:
        with TemporaryDirectory() as tmp:
            masks = Path(tmp)
            _write_mask(masks, "K", 1, slice(8, 24), slice(8, 24))
            _write_mask(masks, "K", 2, slice(2, 6), slice(2, 6))
            circles = [
                CircleAnnotation(3, 26.0, 26.0, 3.0),
                # Has a polygon mask elsewhere in the tile (loaded below), so it must not be added twice.
                CircleAnnotation(2, 4.0, 4.0, 5.0, has_polygon_mask=True, source="inscribed"),
                # The target's own circle never counts.
                CircleAnnotation(1, 16.0, 16.0, 5.0),
            ]
            regions = collect_negative_regions("K", 1, [1, 2], circles, masks, SHAPE)
            self.assertEqual(
                [(r.location_id, r.kind) for r in regions],
                [(2, KIND_POLYGON), (3, KIND_CIRCLE)],
            )
            canvas = composite_negative_mask("K", 1, [1, 2], circles, masks, SHAPE)
            self.assertEqual(canvas.dtype, np.int8)
            self.assertEqual(set(np.unique(canvas)), {-1, 0})
            self.assertEqual(int(canvas[3, 3]), -1)
            self.assertEqual(int(canvas[26, 26]), -1)
            self.assertEqual(int(canvas[16, 16]), 0)

    def test_missing_neighbor_file_is_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            regions = collect_negative_regions("K", 1, [2], [], Path(tmp), SHAPE)
            self.assertEqual(regions, [])

    def test_mask_of_other_size_is_resized(self) -> None:
        with TemporaryDirectory() as tmp:
            masks = Path(tmp)
            small = np.zeros((16, 16), dtype=np.uint8)
            small[4:8, 4:8] = 255
            write_png(masks / "K_5.png", small)
            regions = collect_negative_regions("K", 1, [5], [], masks, SHAPE)
            self.assertEqual(regions[0].mask.shape, SHAPE)
            self.assertTrue(regions[0].mask[10, 10])
            self.assertFalse(regions[0].mask[2, 2])

    def test_combine_signed_resolves_overlap_to_unknown(self) -> None:
        foreground = np.zeros((4, 4), dtype=np.int8)
        foreground[:2, :] = 1
        negative = np.zeros((4, 4), dtype=np.int8)
        negative[1:3, :] = -1
        signed = combine_signed(foreground, negative)
        self.assertEqual(signed[0, 0], 1)
        self.assertEqual(signed[1, 0], 0)  # +1 and -1 conflict
        self.assertEqual(signed[2, 0], -1)
        self.assertEqual(signed[3, 0], 0)
        self.assertTrue(np.array_equal(combine_signed(None, negative), negative))
        self.assertIsNot(combine_signed(None, negative), negative)

    def test_stamp_negative_regions(self) -> None:
        with TemporaryDirectory() as tmp:
            masks = Path(tmp)
            _write_mask(masks, "K", 2, slice(0, 4), slice(0, 4))
            regions = collect_negative_regions("K", 1, [2], [], masks, SHAPE)
            canvas = stamp_negative_regions(regions, SHAPE)
            self.assertEqual(int(canvas.sum()), -16)


class MaskCacheTests(unittest.TestCase):
    def test_hits_do_not_reread_and_oldest_is_evicted(self) -> None:
        with TemporaryDirectory() as tmp:
            masks = Path(tmp)
            for i in range(3):
                _write_mask(masks, "K", i, slice(0, 2), slice(0, 2))
            cache = MaskCache(max_entries=2)
            first = cache.get(masks / "K_0.png")
            self.assertIs(cache.get(masks / "K_0.png"), first)
            cache.get(masks / "K_1.png")
            cache.get(masks / "K_2.png")  # evicts K_0
            (masks / "K_0.png").unlink()
            with self.assertRaises(FileNotFoundError):
                cache.get(masks / "K_0.png")

    def test_zero_size_cache_never_stores(self) -> None:
        with TemporaryDirectory() as tmp:
            masks = Path(tmp)
            _write_mask(masks, "K", 0, slice(0, 2), slice(0, 2))
            cache = MaskCache(max_entries=0)
            self.assertIsNot(cache.get(masks / "K_0.png"), cache.get(masks / "K_0.png"))


if __name__ == "__main__":
    unittest.main()
