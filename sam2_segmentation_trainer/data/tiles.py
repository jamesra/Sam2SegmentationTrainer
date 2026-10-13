"""Per-tile context for mixed-prompt training: which other annotations share each ``imageKey``.

Built once from the usable example list (every location with a raster mask) and each volume's
``circle_annotations.json``. Lookups are O(1) per sample; the structures are small (a few integers per
annotation) and are shared with DataLoader workers by fork/pickle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from annotation_crops.negatives import CircleAnnotation, load_circle_annotations

from sam2_segmentation_trainer.data.manifest import CropExample, load_ignore_ids

TileKey = tuple[str, str]
"""``(volume, image_key)``."""

_MIN_CIRCLE_TARGET_RADIUS_PX = 4.0
"""Circle-only targets smaller than this are noise, not cells."""


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


def crops_dir_of(example: CropExample) -> Path:
    """The ``AnnotationCrops`` folder an example's files live in (``masks/`` is directly under it)."""
    return example.mask_path.parent.parent


def build_tile_context(examples: Sequence[CropExample], *, load_circles: bool = True) -> TileContext:
    """Index ``examples`` by tile and read each volume's circle file.

    ``examples`` should be the whole usable inventory, not just the training split: a validation location is
    still an annotation that a training tile must treat as background.
    """
    by_tile: dict[TileKey, list[int]] = {}
    crops_dirs: dict[str, Path] = {}
    for example in examples:
        by_tile.setdefault((example.volume, example.image_key), []).append(example.location_id)
        crops_dirs.setdefault(example.volume, crops_dir_of(example))
    neighbors = {key: tuple(sorted(set(ids))) for key, ids in by_tile.items()}
    circles: dict[TileKey, tuple[CircleAnnotation, ...]] = {}
    if load_circles:
        for volume, crops in crops_dirs.items():
            for image_key, entries in load_circle_annotations(crops).items():
                circles[(volume, image_key)] = entries
    return TileContext(neighbors=neighbors, circles=circles)


def circle_target_examples(
    examples: Sequence[CropExample],
    context: TileContext,
    *,
    max_fraction: float,
    seed: int,
) -> list[CropExample]:
    """Training examples for circle annotations that have no mask of their own.

    The target is the circle's disc, so these only make sense for the negative-only prompt, where the model
    must find the object from the other annotations and the image alone. A circle qualifies when its location
    is still a circle (``source == "annotated"``), has no mask in the tile, is not on the volume's ignore list,
    has a usable radius and a center inside the tile. At most ``max_fraction`` of ``len(examples)`` are
    returned, chosen with ``seed``, so these samples cannot swamp the real polygon targets.
    """
    templates: dict[TileKey, CropExample] = {}
    for example in examples:
        templates.setdefault((example.volume, example.image_key), example)
    ignored: dict[str, set[int]] = {}
    candidates: list[CropExample] = []
    for key, circles in context.circles.items():
        template = templates.get(key)
        if template is None or template.width is None or template.height is None:
            continue
        crops = crops_dir_of(template)
        if template.volume not in ignored:
            ignored[template.volume] = load_ignore_ids(crops)
        masked = set(context.neighbor_ids(key))
        for circle in circles:
            if (
                circle.has_polygon_mask
                or circle.source != "annotated"
                or circle.location_id in masked
                or circle.location_id in ignored[template.volume]
                or circle.radius < _MIN_CIRCLE_TARGET_RADIUS_PX
                or not circle.center_in((template.height, template.width))
            ):
                continue
            candidates.append(
                replace(
                    template,
                    location_id=circle.location_id,
                    mask_path=template.mask_path.parent / f"{template.image_key}_{circle.location_id}.png",
                    split_key=f"{template.volume}:{circle.location_id}",
                    target_circle=(circle.center_x, circle.center_y, circle.radius),
                )
            )
    limit = int(max(0.0, max_fraction) * len(examples))
    if len(candidates) > limit:
        candidates = random.Random(seed).sample(candidates, limit)
    return sorted(candidates, key=lambda ex: (ex.volume, ex.image_key, ex.location_id))
