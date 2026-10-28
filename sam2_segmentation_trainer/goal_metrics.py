"""Validation metrics for the project goal: identify cells and their connectivity (docs/mask_refinement_plan.md, 6).

The goal is to capture at least 97.5% of each cell's interior with no merges into neighbors, not to trace the
membrane to the pixel. Traced edges are about 9 px (20 nm) from the membrane on average, so every metric ignores a
band of ``b`` px (``val.core_band_px``, 10) around the traced edge:

* **core recall**: the share of ground-truth pixels at least ``b`` inside the edge that are predicted;
* **leak**: the share of predicted pixels inside a neighbor's core (at least ``b`` inside the neighbor);
* **merge**: the sample's prediction covers at least ``merge_share`` (5%) of any one neighbor's visible core and at
  least ``merge_min_px`` (500) px of it, so a stray blob into a neighbor clipped by the tile edge is not a merge;
* **spill**: predicted area more than ``b`` outside the ground truth, as a share of the ground-truth area (it
  catches leaks into cells nobody annotated, which leak cannot see);
* **fragmentation**: the prediction has more connected components (of at least ``min_component`` px) than the
  ground truth.

:func:`sample_goal_metrics` scores one sample; :func:`summarize_goal` pools samples. Only numpy and scipy are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import label

from annotation_crops.prompt_encoding import edge_distance, window
from sam2_segmentation_trainer.data.refine import grow_mask

CORE_BAND_PX = 10.0
MERGE_SHARE = 0.05
MERGE_MIN_PX = 500
"""Pixels of one neighbor's core a prediction must cover before it counts as a merge (``val.merge_min_px``)."""
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
    depth = edge_distance(gt) if has else None
    return {
        "core": (depth >= band) if has else np.zeros(gt.shape, dtype=bool),
        "grown": grow_mask(gt, band) if has else np.zeros(gt.shape, dtype=bool),
        "components": _components(gt, min_component),
        "area": int(gt.sum()),
        "depth": float(depth.max()) if has else 0.0,
    }


def sample_goal_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    neighbor_core: np.ndarray | None = None,
    *,
    band: float = CORE_BAND_PX,
    merge_share: float = MERGE_SHARE,
    merge_min_px: int = MERGE_MIN_PX,
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
            needed = np.maximum(merge_share * totals[present], float(merge_min_px))
            merge = bool(np.any(covered[present] >= needed))
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


DEPTH_BINS = (("lt20", 0.0, 20.0), ("20_60", 20.0, 60.0), ("ge60", 60.0, float("inf")))
"""Bins of a ground truth's maximum depth (px from its edge): thin processes, medium cells, large cells. A profile
under about 20 px wide has a maximum depth under 10 and no core at all."""


@dataclass(slots=True)
class DepthBin:
    """Core recall of the samples whose ground truth has a maximum depth in one :data:`DEPTH_BINS` bin."""

    n: int = 0
    with_core: int = 0
    """Samples that have a core (the others have no recall)."""
    core_recall: float = 0.0
    """Pooled over the bin's core pixels."""
    mean_recall: float = 0.0
    """The mean of the per-sample core recall: every sample counts the same, thin or large."""
    p10: float = 0.0


def summarize_by_depth(records: Sequence[dict[str, Any]]) -> dict[str, DepthBin]:
    """Per-sample core recall stratified by ``gt_depth``; pooled recall is dominated by large cells."""
    out: dict[str, DepthBin] = {}
    for name, low, high in DEPTH_BINS:
        rows = [r for r in records if r.get("gt_depth") is not None and low <= r["gt_depth"] < high]
        summary = DepthBin(n=len(rows))
        recalls = [r["recall"] for r in rows if r.get("recall") is not None]
        summary.with_core = len(recalls)
        if recalls:
            summary.core_recall = sum(r["core_hit"] for r in rows) / max(sum(r["core_gt"] for r in rows), 1)
            summary.mean_recall = float(np.mean(recalls))
            summary.p10 = float(np.percentile(recalls, 10))
        out[name] = summary
    return out


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


SELECTION_METRICS: dict[str, int] = {"core_recall": 1, "iou": 1, "spill": -1, "merge_rate": -1, "frag": -1}
"""The primary family's metrics that select ``best_model.pt`` and drive early stopping, with the direction that is
better (+1 higher, -1 lower). ``iou`` is the tolerant IoU of ``validation.ModeResult``; the rest are
:class:`GoalSummary` fields. The core-recall target is not a gate: the validation masks are traced annotations, not
a checked validation set, so no single threshold on them decides."""

SELECTION_FLOORS: dict[str, float] = {
    "core_recall": 0.004,
    "iou": 0.006,
    "spill": 0.006,
    "merge_rate": 0.002,
    "frag": 0.012,
}
"""Default noise floor of each selection metric (``training.early_stopping_floors``): the 95th-percentile error of a
10% subset of validation set v5 for circle_annotator, measured 2026-10-20 on v5's epoch-0 checkpoint over 200 random
subsets (scripts/valset_size_study.py). The merge rate's measured error was 0.0008 at that model's low rate; 0.002
leaves room for a model that merges more often."""

_EPS = 1e-12


@dataclass(slots=True)
class SelectionKey:
    """One epoch's selection metrics (:data:`SELECTION_METRICS`), for ``best_model.pt`` and early stopping
    (docs/mask_refinement_plan.md, section 6). Compare keys with :func:`improves`."""

    metrics: dict[str, float]

    def __post_init__(self) -> None:
        missing = [name for name in SELECTION_METRICS if name not in self.metrics]
        if missing:
            raise ValueError(f"selection key lacks {', '.join(missing)}")
        # Rounded so float noise in the last digits does not count as a change.
        self.metrics = {name: round(float(self.metrics[name]), 4) for name in SELECTION_METRICS}

    def to_dict(self) -> dict[str, float]:
        return dict(self.metrics)

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> "SelectionKey":
        return cls(dict(values))


def selection_key(summary: GoalSummary, iou: float) -> SelectionKey:
    """The key of one epoch from the primary family's pooled goal metrics and its tolerant IoU."""
    return SelectionKey(
        {
            "core_recall": summary.core_recall,
            "iou": iou,
            "spill": summary.spill,
            "merge_rate": summary.merge_rate,
            "frag": summary.frag,
        }
    )


def selection_changes(
    new: SelectionKey, best: SelectionKey, floors: Mapping[str, float] = SELECTION_FLOORS
) -> tuple[list[str], list[str]]:
    """The metrics of ``new`` that are better and worse than ``best`` by at least their noise floor."""
    better, worse = [], []
    for name, sign in SELECTION_METRICS.items():
        gain = sign * (new.metrics[name] - best.metrics[name])
        floor = float(floors.get(name, SELECTION_FLOORS[name]))
        if gain >= floor - _EPS:
            better.append(name)
        elif gain <= -floor + _EPS:
            worse.append(name)
    return better, worse


def improves(
    new: SelectionKey, best: SelectionKey | None, floors: Mapping[str, float] = SELECTION_FLOORS
) -> bool:
    """Whether ``new`` counts as better than ``best``: at least one selection metric is better by its noise floor
    and none is worse by its noise floor. A change inside the floor is noise either way, so it neither counts nor
    blocks. A trade (one metric better, another worse, both beyond their floors) is not an improvement."""
    if best is None:
        return True
    better, worse = selection_changes(new, best, floors)
    return bool(better) and not worse


def describe_selection(
    new: SelectionKey, best: SelectionKey | None, floors: Mapping[str, float] = SELECTION_FLOORS
) -> str:
    """One line for the log: which metrics moved beyond their noise floor against the best epoch."""
    if best is None:
        return "first validated epoch"
    better, worse = selection_changes(new, best, floors)

    def moved(names: list[str]) -> str:
        return ", ".join(f"{n} {best.metrics[n]:.4f} -> {new.metrics[n]:.4f}" for n in names) or "none"

    return f"against the best epoch, better beyond the noise floor: {moved(better)}; worse: {moved(worse)}"
