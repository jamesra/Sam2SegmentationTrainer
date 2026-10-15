"""Per-tile context for mixed-prompt training: which other annotations share each ``imageKey``.

Built once from the usable example list (every location with a raster mask) and each volume's
``circle_annotations.json``. Lookups are O(1) per sample; the structures are small (a few integers per
annotation) and are shared with DataLoader workers by fork/pickle.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from annotation_crops.negatives import (
    SOURCE_ANNOTATED,
    CircleAnnotation,
    load_circle_annotations,
    load_circle_scale,
)

from sam2_segmentation_trainer.data.manifest import CropExample

TileKey = tuple[str, str]
"""``(volume, image_key)``."""


@dataclass(slots=True)
class TileContext:
    """Neighbor and circle lookups for every tile in the training inventory.

    ``neighbors`` maps a tile to the location ids that have a mask PNG in it. ``circles`` maps a tile to the
    circle annotations the exporter wrote for it. Both are read-only after construction.
    """

    neighbors: dict[TileKey, tuple[int, ...]] = field(default_factory=dict)
    circles: dict[TileKey, tuple[CircleAnnotation, ...]] = field(default_factory=dict)

    def neighbor_ids(self, key: TileKey) -> tuple[int, ...]:
        """Location ids with a mask in the tile."""
        return self.neighbors.get(key, ())

    def circles_in(self, key: TileKey) -> tuple[CircleAnnotation, ...]:
        """Circle annotations in the tile."""
        return self.circles.get(key, ())

    def circle_for(self, key: TileKey, location_id: int) -> CircleAnnotation | None:
        """The circle annotation of ``location_id`` in the tile, if the exporter wrote one."""
        for circle in self.circles.get(key, ()):
            if circle.location_id == location_id:
                return circle
        return None

    def adjacent_shape(self, key: TileKey, location_id: int):
        """Boolean raster of the linked polygon from a neighbouring section, or None.

        The exporter does not write ``shapes.jsonl`` yet (plan phase B), so this is always None and adjacent-section
        seeds are not drawn. A loader that has the shapes overrides or replaces this.
        """
        return None


def crops_dir_of(example: CropExample) -> Path:
    """The ``AnnotationCrops`` folder an example's files live in (``masks/`` is directly under it)."""
    return example.mask_path.parent.parent


def check_circle_files(
    crops_dirs: dict[str, Path],
    loaded: dict[str, int],
    *,
    require: bool,
    circle_scale: float | None,
    by_source: dict[str, Counter] | None = None,
) -> list[str]:
    """Report volumes whose circle file is missing or empty, and how each file's radii are used.

    ``loaded`` maps each volume to its number of circle entries; ``by_source`` optionally to its entries per
    ``source``. With ``require`` a missing or empty file raises, because circle-mask prompts would silently
    disappear and the prompt mix would shift to the rest. Returns the warnings it printed.
    """
    missing = sorted(volume for volume in crops_dirs if loaded.get(volume, 0) == 0)
    if missing and require:
        raise RuntimeError(
            f"circle_annotations.json is missing or empty for {missing}, so circle-mask prompts are unavailable. "
            "Export circles for these volumes, or set data.require_circle_file: false to train without them."
        )
    messages = [f"no circle entries for {volume}" for volume in missing]
    for volume, crops in sorted(crops_dirs.items()):
        if volume in missing:
            continue
        scale = load_circle_scale(crops)
        counts = (by_source or {}).get(volume, Counter())
        annotated = counts.get(SOURCE_ANNOTATED, 0)
        inscribed = loaded[volume] - annotated
        seed_note = "as written" if circle_scale is None else f"x{circle_scale / (scale or 1.0):.4g} (diameter scale {circle_scale})"
        print(
            f"circles {volume}: {loaded[volume]} entries, file diameter_scale {scale}; "
            f"{annotated} annotated at true radius (trust fractions apply, no diameter scale), "
            f"{inscribed} inscribed radii {seed_note}"
        )
    for message in messages:
        print(f"WARNING: {message}")
    return messages


def build_tile_context(
    examples: Sequence[CropExample],
    *,
    load_circles: bool = True,
    require_circles: bool = False,
    circle_scale: float | None = None,
) -> TileContext:
    """Index ``examples`` by tile and read each volume's circle file.

    ``examples`` should be the whole usable inventory, not just the training split: a validation location is
    still an annotation that a training tile must treat as background. ``circle_scale`` is the diameter scale
    of the seed circles the caller draws: inscribed radii are rescaled from the file's own ``diameter_scale`` to
    it (None keeps them as written). Annotated circles always load at their true radius. ``require_circles`` turns a missing file into an error (:func:`check_circle_files`).
    """
    by_tile: dict[TileKey, list[int]] = {}
    crops_dirs: dict[str, Path] = {}
    for example in examples:
        by_tile.setdefault((example.volume, example.image_key), []).append(example.location_id)
        crops_dirs.setdefault(example.volume, crops_dir_of(example))
    neighbors = {key: tuple(sorted(set(ids))) for key, ids in by_tile.items()}
    circles: dict[TileKey, tuple[CircleAnnotation, ...]] = {}
    if load_circles:
        loaded: dict[str, int] = {}
        by_source: dict[str, Counter] = {}
        for volume, crops in crops_dirs.items():
            entries_by_key = load_circle_annotations(crops, circle_scale)
            loaded[volume] = sum(len(entries) for entries in entries_by_key.values())
            by_source[volume] = Counter(c.source for entries in entries_by_key.values() for c in entries)
            for image_key, entries in entries_by_key.items():
                circles[(volume, image_key)] = entries
        check_circle_files(
            crops_dirs, loaded, require=require_circles, circle_scale=circle_scale, by_source=by_source
        )
    return TileContext(neighbors=neighbors, circles=circles)
