"""Validation metrics for the project goal: identify cells and their connectivity (docs/mask_refinement_plan.md, 6).

The goal is to capture at least 97.5% of each cell's interior with no merges into neighbors, not to trace the
membrane to the pixel. Traced edges are about 9 px (20 nm) from the membrane on average, so every metric ignores a
band of ``b`` px (``val.core_band_px``, 10) around the traced edge:

* **core recall**: the share of ground-truth pixels at least ``b`` inside the edge that are predicted;
* **leak**: the share of predicted pixels inside a neighbor's core (at least ``b`` inside the neighbor);
* **merge**: the sample's prediction covers at least ``merge_share`` (5%) of any one neighbor's core;
* **spill**: predicted area more than ``b`` outside the ground truth, as a share of the ground-truth area (it
  catches leaks into cells nobody annotated, which leak cannot see);
* **fragmentation**: the prediction has more connected components (of at least ``min_component`` px) than the
  ground truth.

:func:`sample_goal_metrics` scores one sample; :func:`summarize_goal` pools samples. Only numpy and scipy are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import label

from annotation_crops.prompt_encoding import edge_distance, window
from sam2_segmentation_trainer.data.refine import grow_mask

CORE_BAND_PX = 10.0
MERGE_SHARE = 0.05
MIN_COMPONENT_PX = 200
CORE_RECALL_TARGET = 0.975
NO_HARM_RECALL = 0.005
"""A near-correct output passes when its core recall is down by no more than this (0.5 points)..."""
NO_HARM_SPILL = 0.005
"""... and its spill is up by no more than this share of the ground-truth area (0.5%)."""

COUNT_FIELDS = ("core_gt", "core_hit", "pred_area", "leak_px", "gt_area", "spill_px")


def _components(mask: np.ndarray, min_pixels: int) -> int:
    """Connected components of at least ``min_pixels``, labelled inside the mask's bounding box."""
    box = window(mask, 0)
    if box is None:
        return 0
    labels, count = label(mask[box])
    sizes = np.bincount(labels.ravel())[1:]
    return int((sizes >= min_pixels).sum())


def ground_truth_terms(gt: np.ndarray, band: float, min_component: int) -> dict[str, Any]:
    """What :func:`sample_goal_metrics` derives from the ground truth alone: pass the dict as ``cache`` to score
    several predictions (the output and the input mask) against one ground truth without recomputing it."""
    gt = np.asarray(gt).astype(bool)
    has = bool(gt.any())
    return {
        "core": (edge_distance(gt) >= band) if has else np.zeros(gt.shape, dtype=bool),
        "grown": grow_mask(gt, band) if has else np.zeros(gt.shape, dtype=bool),
        "components": _components(gt, min_component),
        "area": int(gt.sum()),
    }


def sample_goal_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    neighbor_core: np.ndarray | None = None,
    *,
    band: float = CORE_BAND_PX,
    merge_share: float = MERGE_SHARE,
    min_component: int = MIN_COMPONENT_PX,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The goal metrics of one prediction.

    ``neighbor_core`` is the int16 map from the dataset (``i + 1`` on neighbor ``i``'s core, 0 elsewhere); None
    means the tile has no other annotation. The result holds the raw pixel counts (so samples can be pooled) and
    the per-sample flags ``merge`` and ``frag``; ``recall`` is None when the ground truth has no core. ``cache`` is
    a dict from :func:`ground_truth_terms` (it is filled in when empty), shared between predictions of one
    ground truth.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if cache is None:
        cache = {}
    if not cache:
        cache.update(ground_truth_terms(gt, band, min_component))
    core = cache["core"]
    core_gt = int(core.sum())
    core_hit = int((pred & core).sum())
    gt_area = cache["area"]
    pred_area = int(pred.sum())
    spill_px = int((pred & ~cache["grown"]).sum()) if gt_area else pred_area
    leak_px = 0
    merge = False
    if neighbor_core is not None and pred.any():
        labels = np.asarray(neighbor_core)
        inside = labels[pred]
        leak_px = int((inside > 0).sum())
        if leak_px:
            top = int(labels.max())
            covered = np.bincount(inside, minlength=top + 1)[1:]
            totals = np.bincount(labels.ravel(), minlength=top + 1)[1:]
            present = totals > 0
            merge = bool(np.any(covered[present] >= merge_share * totals[present]))
    return {
        "core_gt": core_gt,
        "core_hit": core_hit,
        "recall": (core_hit / core_gt) if core_gt else None,
        "pred_area": pred_area,
        "leak_px": leak_px,
        "gt_area": gt_area,
        "spill_px": spill_px,
        "merge": merge,
        "frag": _components(pred, min_component) > cache["components"],
    }


@dataclass(slots=True)
class GoalSummary:
    """Pooled goal metrics of a set of samples (a validation family, or its Approved subset)."""

    n: int = 0
    core_recall: float = 0.0
    """Pooled over all core pixels (the "interior volume" figure)."""
    core_recall_p10: float = 0.0
    """The 10th percentile of the per-sample core recall (samples that have a core)."""
    core_recall_ok: float = 0.0
    """Share of samples with core recall at or above the target (0.975)."""
    leak: float = 0.0
    merge_rate: float = 0.0
    spill: float = 0.0
    frag: float = 0.0


def summarize_goal(
    records: Sequence[dict[str, Any]], *, prefix: str = "", target: float = CORE_RECALL_TARGET
) -> GoalSummary:
    """Pool ``records`` carrying ``{prefix}core_gt`` etc. (``prefix`` is "" for the output, "input_" for the input)."""
    rows = [r for r in records if f"{prefix}core_gt" in r]
    summary = GoalSummary(n=len(rows))
    if not rows:
        return summary

    def total(name: str) -> int:
        return int(sum(r[f"{prefix}{name}"] for r in rows))

    summary.core_recall = total("core_hit") / max(total("core_gt"), 1)
    recalls = [r[f"{prefix}recall"] for r in rows if r[f"{prefix}recall"] is not None]
    if recalls:
        summary.core_recall_p10 = float(np.percentile(recalls, 10))
        summary.core_recall_ok = float(np.mean([v >= target for v in recalls]))
    summary.leak = total("leak_px") / max(total("pred_area"), 1)
    summary.merge_rate = float(np.mean([bool(r[f"{prefix}merge"]) for r in rows]))
    summary.spill = total("spill_px") / max(total("gt_area"), 1)
    summary.frag = float(np.mean([bool(r[f"{prefix}frag"]) for r in rows]))
    return summary


def add_goal_fields(record: dict[str, Any], metrics: dict[str, Any], prefix: str = "") -> None:
    """Copy :func:`sample_goal_metrics` output into a record, with ``prefix`` on every key."""
    for key, value in metrics.items():
        record[f"{prefix}{key}"] = value


def sample_no_harm(record: dict[str, Any]) -> bool:
    """Whether the output is no worse than the input: core recall down at most 0.5 points and spill up at most 0.5%.

    A sample without a core (no recall) passes the recall half. Spill is compared as a share of the sample's own
    ground-truth area.
    """
    out_recall, in_recall = record.get("recall"), record.get("input_recall")
    recall_ok = out_recall is None or in_recall is None or out_recall >= in_recall - NO_HARM_RECALL
    area = max(int(record["gt_area"]), 1)
    spill_ok = record["spill_px"] / area <= record["input_spill_px"] / area + NO_HARM_SPILL
    return bool(recall_ok and spill_ok)


@dataclass(slots=True)
class SelectionKey:
    """Orders epochs for ``best_model.pt`` and early stopping (docs/mask_refinement_plan.md, section 6).

    Among epochs whose pooled core recall reaches the target, the lowest merge rate wins, with spill breaking
    ties. If none reaches the target, the highest core recall wins. Every epoch that reaches the target beats
    every epoch that does not. Compare keys with ``>``.
    """

    gated: bool
    primary: float
    secondary: float = 0.0
    values: tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        # Rounded so float noise in the last digits does not count as an improvement and reset the patience.
        self.values = (1.0 if self.gated else 0.0, round(self.primary, 4), round(self.secondary, 4))

    def __gt__(self, other: "SelectionKey") -> bool:
        return self.values > other.values

    def to_list(self) -> list[float]:
        return list(self.values)

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "SelectionKey":
        gate, primary, secondary = (float(v) for v in values)
        return cls(gated=gate >= 0.5, primary=primary, secondary=secondary)


def selection_key(summary: GoalSummary, target: float = CORE_RECALL_TARGET) -> SelectionKey:
    """The key of one epoch from the primary family's pooled metrics."""
    if summary.core_recall >= target:
        return SelectionKey(True, -summary.merge_rate, -summary.spill)
    return SelectionKey(False, summary.core_recall, 0.0)


EARLY_STOPPING_MIN_DELTA = 0.002
"""Default noise floor of the merge rate and spill in :func:`improves` (``training.early_stopping_min_delta``)."""

RECALL_MIN_DELTA = 0.0005
"""Core recall below the gate must improve by at least this to count as better."""

_EPS = 1e-12


def improves(
    new: SelectionKey,
    best: SelectionKey | None,
    min_delta: float = EARLY_STOPPING_MIN_DELTA,
    recall_delta: float = RECALL_MIN_DELTA,
) -> bool:
    """Whether ``new`` counts as better than ``best`` once the noise floor is applied.

    Compare with :func:`selection_key`'s ordering, but a change smaller than the noise floor is not an
    improvement and does not reset the patience:

    * reaching the core-recall gate beats not reaching it, whatever the other numbers;
    * both at the gate: the merge rate must fall by at least ``min_delta``; if it moves by less than that either
      way, the spill must fall by at least ``min_delta``;
    * both below the gate: the core recall must rise by at least ``recall_delta``.
    """
    if best is None:
        return True
    if new.gated != best.gated:
        return new.gated
    if new.gated:
        merge_gain = new.primary - best.primary  # the keys store the negated merge rate
        if merge_gain >= min_delta - _EPS:
            return True
        if merge_gain <= -min_delta + _EPS:
            return False
        return (new.secondary - best.secondary) >= min_delta - _EPS
    return (new.primary - best.primary) >= recall_delta - _EPS
