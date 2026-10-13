"""z-slab train/val split: evenly spaced validation slabs with guard sections."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sam2_segmentation_trainer.data.manifest import CropExample
from sam2_segmentation_trainer.data.splits import (
    include_new_keys,
    load_split,
    make_zslab_split,
    save_split,
)


def _examples(volume: str, zs, per_section: int = 10, start_id: int = 0) -> list[CropExample]:
    out: list[CropExample] = []
    loc = start_id
    for z in zs:
        for _ in range(per_section):
            out.append(
                CropExample(
                    volume=volume, z=z, downsample=1, image_key=f"{volume}_{z}_k", location_id=loc,
                    image_path=Path("i.png"), json_path=Path("i.json"), mask_path=Path("m.png"),
                    split_key=f"{volume}:{loc}",
                )
            )
            loc += 1
    return out


def _runs(zs) -> list[tuple[int, int]]:
    ordered = sorted(zs)
    runs, start, prev = [], ordered[0], ordered[0]
    for z in ordered[1:]:
        if z != prev + 1:
            runs.append((start, prev))
            start = z
        prev = z
    runs.append((start, prev))
    return runs


class ZSlabSplitTests(unittest.TestCase):
    def test_five_even_slabs_total_fraction_and_guards(self) -> None:
        examples = _examples("V", range(100, 300))  # 200 sections x 10
        split = make_zslab_split(examples, val_fraction=0.10, slab_count=5, guard_sections=1)
        slabs = _runs(split.val_z["V"])
        self.assertEqual(len(slabs), 5)
        self.assertAlmostEqual(len(split.val_keys) / len(examples), 0.10, delta=0.02)
        centers = [(lo + hi) / 2 for lo, hi in slabs]
        for got, want in zip(centers, [120, 160, 200, 240, 280]):
            self.assertLessEqual(abs(got - want), 3)
        guard = set(split.guard_z["V"])
        for lo, hi in slabs:
            self.assertIn(lo - 1, guard)
            self.assertIn(hi + 1, guard)
        self.assertEqual(len(guard), 10)
        train, val, used = set(split.train_keys), set(split.val_keys), set()
        self.assertFalse(train & val)
        by_key = {ex.split_key: ex for ex in examples}
        for key in train:
            self.assertNotIn(by_key[key].z, guard)
            self.assertNotIn(by_key[key].z, set(split.val_z["V"]))
        self.assertEqual(len(train) + len(val) + 10 * len(guard), len(examples))

    def test_every_section_lands_on_one_side(self) -> None:
        examples = _examples("V", range(0, 60), per_section=3)
        split = make_zslab_split(examples, val_fraction=0.1, slab_count=5, guard_sections=1)
        side = {}
        for key in split.train_keys:
            side[key] = "t"
        for key in split.val_keys:
            side[key] = "v"
        by_z: dict[int, set[str]] = {}
        for ex in examples:
            by_z.setdefault(ex.z, set()).add(side.get(ex.split_key, "g"))
        self.assertTrue(all(len(sides) == 1 for sides in by_z.values()))

    def test_gaps_and_uneven_density_follow_examples(self) -> None:
        dense = _examples("V", range(0, 50), per_section=20)
        sparse = _examples("V", range(200, 400, 4), per_section=2, start_id=10_000)
        examples = dense + sparse
        split = make_zslab_split(examples, val_fraction=0.1, slab_count=5, guard_sections=1)
        self.assertAlmostEqual(len(split.val_keys) / len(examples), 0.10, delta=0.03)
        self.assertEqual(len(_runs(split.val_z["V"])) >= 1, True)

    def test_volumes_are_split_independently(self) -> None:
        examples = _examples("A", range(0, 100)) + _examples("B", range(500, 1000), start_id=10_000)
        split = make_zslab_split(examples, val_fraction=0.1)
        self.assertEqual(set(split.val_z), {"A", "B"})
        for volume, n in (("A", 1000), ("B", 5000)):
            n_val = sum(1 for key in split.val_keys if key.startswith(volume + ":"))
            self.assertAlmostEqual(n_val / n, 0.1, delta=0.02)

    def test_save_load_and_new_keys_follow_sections(self) -> None:
        examples = _examples("V", range(0, 100))
        split = make_zslab_split(examples, val_fraction=0.1)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            save_split(split, path)
            loaded = load_split(path)
        self.assertEqual(loaded, split)
        self.assertEqual(loaded.mode, "zslab")
        val_z = loaded.val_z["V"][0]
        guard_z = loaded.guard_z["V"][0]
        fresh = [
            _examples("V", [val_z], per_section=1, start_id=90_000)[0],
            _examples("V", [guard_z], per_section=1, start_id=90_001)[0],
            _examples("V", [99], per_section=1, start_id=90_002)[0],
        ]
        grown = include_new_keys(loaded, examples + fresh, train_fraction=0.9, seed=1)
        self.assertIn("V:90000", grown.val_keys)
        self.assertNotIn("V:90001", grown.val_keys + grown.train_keys)
        self.assertIn("V:90002", grown.train_keys)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            make_zslab_split(_examples("V", range(10)), val_fraction=0.0)
        with self.assertRaises(ValueError):
            make_zslab_split(_examples("V", range(10)), slab_count=0)


if __name__ == "__main__":
    unittest.main()
