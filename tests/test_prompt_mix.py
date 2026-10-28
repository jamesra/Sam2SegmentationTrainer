"""The prompt mix: every variant of a sample, the first included, is drawn from the ``prob_*`` shares without
replacement (no family is forced on a sample), with the three circle peers as ordinary entries."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from sam2_segmentation_trainer.data.dataset import EMSegDataset
from sam2_segmentation_trainer.data.prompts import CIRCLE_PROMPT_TYPES, FULL_TILE_PROMPTS, PromptConfig, PromptType

PLANS = 6000


def simulate(cfg: PromptConfig, samples: int, *, can_adjacent: bool, allowed=None, seed: int = 123) -> Counter:
    """Per-family variant counts of ``samples`` independent without-replacement draws of the mix (the same rule as
    ``choose_prompt_type``, written out again: the adjacent share moves to the circle peers in proportion when it is
    unavailable, and each draw renormalises over the families left)."""
    rng = np.random.default_rng(seed)
    base = dict(cfg.probabilities())
    if not can_adjacent:
        moved = base.pop(PromptType.ADJACENT_POLY_SEED)
        circles = sum(base[k] for k in CIRCLE_PROMPT_TYPES)
        for kind in CIRCLE_PROMPT_TYPES:
            base[kind] += moved * base[kind] / circles
    if allowed is not None:
        base = {k: v for k, v in base.items() if k in allowed}
    counts: Counter = Counter()
    for _ in range(samples):
        left = dict(base)
        for _k in range(min(cfg.variants_per_sample, len(left))):
            kinds = list(left)
            weights = np.array([left[k] for k in kinds])
            kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
            counts[kind] += 1
            del left[kind]
    return counts


def shares(counts: Counter) -> dict:
    total = sum(counts.values())
    return {kind: counts[kind] / total for kind in counts}


class PromptMixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PromptConfig()
        self.dataset = EMSegDataset([], split="train", image_size=32, prompt_cfg=self.cfg, seed=11)

    def plans(self, **kwargs) -> list[list[PromptType]]:
        return [self.dataset._plan_types(self.cfg, i, 0, forced=None, **kwargs) for i in range(PLANS)]

    def test_the_default_mix_has_ten_peer_entries_summing_to_one(self) -> None:
        probs = self.cfg.probabilities()
        self.assertEqual(len(probs), 10)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(
            [probs[k] for k in CIRCLE_PROMPT_TYPES], [0.10, 0.20, 0.05]
        )

    def test_a_sample_gets_four_distinct_families(self) -> None:
        for plan in self.plans(full_tile=False, has_adjacent=True)[:500]:
            self.assertEqual(len(plan), 4)
            self.assertEqual(len(set(plan)), 4)

    def test_no_family_is_forced_on_the_first_variant(self) -> None:
        first = Counter(plan[0] for plan in self.plans(full_tile=False, has_adjacent=True))
        for kind, share in self.cfg.probabilities().items():
            self.assertAlmostEqual(first[kind] / PLANS, share, delta=0.015, msg=kind)

    def test_realised_shares_match_the_without_replacement_expectation(self) -> None:
        for has_adjacent in (False, True):
            realised = shares(Counter(k for plan in self.plans(full_tile=False, has_adjacent=has_adjacent) for k in plan))
            expected = shares(simulate(self.cfg, 20000, can_adjacent=has_adjacent))
            self.assertEqual(set(realised), set(expected))
            for kind in expected:
                self.assertAlmostEqual(realised[kind], expected[kind], delta=0.01, msg=(has_adjacent, kind))
            # Without replacement a rare family is over-represented relative to its prob_* share of 1.0.
            self.assertGreater(realised[PromptType.CIRCLE_PULL_THROUGH], self.cfg.prob_circle_pull_through)

    def test_a_full_tile_draws_points_and_the_three_circle_peers(self) -> None:
        plans = self.plans(full_tile=True, has_adjacent=True)
        for plan in plans[:300]:
            self.assertEqual(set(plan), set(FULL_TILE_PROMPTS))
        first = shares(Counter(plan[0] for plan in plans))
        expected = shares(simulate(PromptConfig(variants_per_sample=1), 20000, can_adjacent=False, allowed=FULL_TILE_PROMPTS))
        for kind in FULL_TILE_PROMPTS:
            self.assertAlmostEqual(first[kind], expected[kind], delta=0.02, msg=kind)


if __name__ == "__main__":
    unittest.main()
