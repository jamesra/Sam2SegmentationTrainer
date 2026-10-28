"""What the training loop logs about the prompt mix, and the start-up line about adjacent-seed placement."""

from __future__ import annotations

import contextlib
import io
import unittest
from collections import Counter

from sam2_segmentation_trainer.data.tiles import TileContext, adjacent_placement_summary
from sam2_segmentation_trainer.train import PROMPT_DROPPED, _log_prompt_fallbacks, _log_prompt_mix


class FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def by_tag(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for tag, value, _step in self.scalars:
            out.setdefault(tag, []).append(value)
        return out


class FallbackLogTests(unittest.TestCase):
    def test_one_value_per_tag_holding_the_total_asked(self) -> None:
        writer = FakeWriter()
        counts = Counter(
            {
                ("adjacent_poly_seed", "circle_seed"): 5,
                ("adjacent_poly_seed", "points"): 3,
                ("adjacent_poly_seed", PROMPT_DROPPED): 2,
                ("circle_annotator", "circle_seed"): 4,
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            _log_prompt_fallbacks(writer, 7, counts)
        tags = writer.by_tag()
        # Each tag is written once per epoch: TensorBoard would show only one of several values at one step.
        self.assertTrue(all(len(values) == 1 for values in tags.values()), tags)
        self.assertEqual(tags["train/prompt_fallback/adjacent_poly_seed"], [10.0])
        self.assertEqual(tags["train/prompt_fallback/circle_annotator"], [4.0])
        self.assertEqual(tags["train/prompt_fallback/adjacent_poly_seed/points"], [3.0])
        self.assertEqual(tags["train/prompt_fallback/adjacent_poly_seed/dropped"], [2.0])
        self.assertTrue(all(step == 7 for _tag, _value, step in writer.scalars))

    def test_the_mix_line_counts_variants(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _log_prompt_mix(FakeWriter(), 0, Counter({"points": 3, "circle_seed": 1}))
        self.assertIn("prompt mix (4 variants)", out.getvalue())


class PlacementSummaryTests(unittest.TestCase):
    def test_one_line_names_projected_fallback_and_missing_volumes(self) -> None:
        context = TileContext(
            shapes={"RC1": object(), "RC2": object(), "RPC1": object()},
            projection={
                "RC1": {"placement": "centroid", "reason": "no data.registration_root"},
                "RC2": {"placement": "centroid", "reason": "no data.registration_root"},
                "RPC1": {"placement": "projected"},
            },
        )
        line = adjacent_placement_summary(context, ["RPC2", "RC1", "RC2", "RPC1"])
        self.assertNotIn("\n", line)
        self.assertTrue(
            line.startswith(
                "adjacent_poly_seed: projected for [RPC1] / centroid fallback for [RC1, RC2] / no shapes.jsonl for [RPC2]"
            ),
            line,
        )
        self.assertIn("no data.registration_root", line)
        self.assertIn("data.registration_root and data.corrections_root", line)

    def test_all_projected_needs_no_hint(self) -> None:
        context = TileContext(shapes={"RPC1": object()}, projection={"RPC1": {"placement": "projected"}})
        self.assertEqual(
            adjacent_placement_summary(context, ["RPC1"]),
            "adjacent_poly_seed: projected for [RPC1] / centroid fallback for []",
        )

    def test_a_configured_placement_is_named(self) -> None:
        context = TileContext(shapes={"RC1": object()}, adjacent_placement="centroid")
        self.assertEqual(
            adjacent_placement_summary(context, ["RC1"]), "adjacent_poly_seed: centroid placement (configured) for [RC1]"
        )


if __name__ == "__main__":
    unittest.main()
