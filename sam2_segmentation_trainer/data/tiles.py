"""Per-tile context for mixed-prompt training: which other annotations share each ``imageKey``.

Built once from the usable example list (every location with a raster mask), each volume's
``circle_annotations.json`` and each volume's ``shapes.jsonl`` (the exporter's outlines and adjacent-section links,
:mod:`sam2_segmentation_trainer.data.shapes`), whose neighbour outlines are carried into their targets' mosaic frames
when the volume's registration data is configured (:mod:`sam2_segmentation_trainer.data.projection`). Lookups are
O(1) per sample; the structures are small (a few integers per annotation, the outlines packed in flat numpy arrays)
and are shared with DataLoader workers by fork/pickle; no worker reads a transform.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from annotation_crops.negatives import (
    SOURCE_ANNOTATED,
    CircleAnnotation,
    load_circle_annotations,
    load_circle_scale,
)

from sam2_segmentation_trainer.data.manifest import CropExample
from sam2_segmentation_trainer.data.projection import (
    RESIDUAL_AUTO,
    RESIDUAL_MODES,
    SectionProjector,
    project_volume_shapes,
    projector_for_volume,
)
from sam2_segmentation_trainer.data.shapes import (
    ADJACENT_PLACEMENTS,
    LINK_MAX_DZ,
    PLACEMENT_CENTROID,
    PLACEMENT_PROJECTED,
    SHAPES_FILENAME,
    AdjacentDraw,
    VolumeShapes,
    draw_projected,
    load_volume_shapes,
    pick_neighbour_raster,
)

TileKey = tuple[str, str]
"""``(volume, image_key)``."""


@dataclass(slots=True)
class TileContext:
    """Neighbor and circle lookups for every tile in the training inventory.

    ``neighbors`` maps a tile to the location ids that have a mask PNG in it. ``circles`` maps a tile to the
    circle annotations the exporter wrote for it. ``shapes`` maps a volume to its adjacent-section outlines and links
    (a volume without ``shapes.jsonl`` is absent). All are read-only after construction.
    """

    neighbors: dict[TileKey, tuple[int, ...]] = field(default_factory=dict)
    circles: dict[TileKey, tuple[CircleAnnotation, ...]] = field(default_factory=dict)
    shapes: dict[str, VolumeShapes] = field(default_factory=dict)
    adjacent_placement: str = PLACEMENT_PROJECTED
    """Where the neighbour outline is drawn (:mod:`~sam2_segmentation_trainer.data.shapes`, "Placement"). A volume
    whose outlines were not projected uses the centroid placement."""
    projection: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per volume, how its adjacent outlines are placed and from which registration and correction data (the stos
    group, the VikingXML, the correction set's manifest provenance and digest); recorded in ``training_data.json``."""

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

    def adjacent_shape(
        self,
        key: TileKey,
        location_id: int,
        *,
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):
        """Boolean ``shape`` raster, in the tile's pixel frame, of ONE linked outline of ``location_id`` from a
        section 1-2 away, or None (no ``shapes.jsonl`` for the volume, no linked neighbour with an outline, or none
        that reaches the tile). The adjacent polygon seed is available for a sample exactly when this is not None.

        With several neighbours one is picked uniformly by ``rng``; the dataset passes a generator seeded from
        ``(seed, epoch, index)`` so a resumed epoch replays the same pick (:func:`pick_neighbour_raster`).
        Placement: the exporter's coordinates are unregistered section mosaic pixels (a linked neighbour lies
        thousands of pixels from its target), so by default the outline is translated to put its centroid on the
        target's centroid; ``adjacent_placement="raw"`` draws it as written, the hook for the pending section
        projection. ``fit_adjacent_poly_seed`` re-centres it on the target anyway, so only its shape matters today.
        """
        volume_shapes = self.shapes.get(key[0])
        if volume_shapes is None:
            return None
        placement = PLACEMENT_CENTROID if self.adjacent_placement == PLACEMENT_PROJECTED else self.adjacent_placement
        return pick_neighbour_raster(volume_shapes, key[1], location_id, shape, rng, placement)

    def adjacent_draw(
        self,
        key: TileKey,
        location_id: int,
        *,
        target: np.ndarray,
        rng: np.random.Generator | None = None,
        overlap_min: float = 0.6,
        erode_px: tuple[float, float] = (0.0, 0.0),
    ) -> AdjacentDraw | None:
        """The adjacent-section outline for one sample (before augmentation), or None when the family is unavailable.

        With the projected placement and a projected volume: :func:`draw_projected` (the outline where the
        registration carries it, eroded, with the overlap rule against ``target``). Otherwise :meth:`adjacent_shape`
        (centroid placement), returned unprojected for ``fit_adjacent_poly_seed`` to fit inside the cell.
        """
        volume_shapes = self.shapes.get(key[0])
        if (
            self.adjacent_placement == PLACEMENT_PROJECTED
            and volume_shapes is not None
            and volume_shapes.projected is not None
        ):
            return draw_projected(
                volume_shapes, key[1], location_id, target, rng, overlap_min=overlap_min, erode_px=erode_px
            )
        shape = (int(target.shape[0]), int(target.shape[1]))
        raster = self.adjacent_shape(key, location_id, shape=shape, rng=rng)
        return None if raster is None else AdjacentDraw(raster=raster, projected=False)


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
    load_shapes: bool = True,
    adjacent_placement: str = PLACEMENT_PROJECTED,
    registration_root: Path | str | None = None,
    corrections_root: Path | str | None = None,
    adjacent_residual: str = RESIDUAL_AUTO,
) -> TileContext:
    """Index ``examples`` by tile and read each volume's circle file and ``shapes.jsonl``.

    ``examples`` should be the whole usable inventory, not just the training split: a validation location is
    still an annotation that a training tile must treat as background. ``circle_scale`` is the diameter scale
    of the seed circles the caller draws: inscribed radii are rescaled from the file's own ``diameter_scale`` to
    it (None keeps them as written). Annotated circles always load at their true radius. ``require_circles`` turns a missing file into an error (:func:`check_circle_files`).

    ``load_shapes`` reads each volume's ``shapes.jsonl`` (:func:`load_volume_shapes`), keeping the links of the
    inventory's locations and the outlines of their neighbours. A missing file prints a warning and leaves that
    volume without adjacent shapes: ``adjacent_shape`` returns None there and the adjacent seed's share moves to the
    circle peers, as it did before the exporter wrote the file.

    ``adjacent_placement="projected"`` (the default) carries each volume's neighbour outlines into their targets'
    frames (:func:`attach_projection`) from ``registration_root`` (``data.registration_root``) and
    ``corrections_root`` (``data.corrections_root``); ``adjacent_residual`` is ``data.adjacent_residual``. A volume
    that cannot be projected (a root is null, its files are missing, or its stos group is chained) uses the centroid
    placement, with one warning per volume.
    """
    if adjacent_placement not in ADJACENT_PLACEMENTS:
        raise ValueError(f"adjacent placement must be one of {ADJACENT_PLACEMENTS}, got {adjacent_placement!r}")
    if adjacent_residual not in RESIDUAL_MODES:
        raise ValueError(f"data.adjacent_residual must be one of {RESIDUAL_MODES}, got {adjacent_residual!r}")
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
    shapes: dict[str, VolumeShapes] = {}
    if load_shapes:
        ids_by_volume: dict[str, set[int]] = {}
        for example in examples:
            ids_by_volume.setdefault(example.volume, set()).add(example.location_id)
        for volume, crops in sorted(crops_dirs.items()):
            loaded_shapes = load_volume_shapes(crops / SHAPES_FILENAME, ids_by_volume.get(volume, set()))
            if loaded_shapes is None:
                print(
                    f"WARNING: no {SHAPES_FILENAME} for {volume} ({crops}); adjacent_poly_seed is not drawn there "
                    "and its share moves to the circle peers"
                )
                continue
            shapes[volume] = loaded_shapes
            print(
                f"shapes {volume}: {len(loaded_shapes.links)} of {len(ids_by_volume.get(volume, ()))} locations have "
                f"a linked outline 1-{LINK_MAX_DZ} sections away ({len(loaded_shapes)} outlines kept)"
            )
    projection: dict[str, dict[str, Any]] = {}
    if adjacent_placement == PLACEMENT_PROJECTED:
        for volume, volume_shapes in sorted(shapes.items()):
            projection[volume] = attach_projection(
                volume,
                volume_shapes,
                registration_root=registration_root,
                corrections_root=corrections_root,
                residual=adjacent_residual,
            )
    return TileContext(
        neighbors=neighbors,
        circles=circles,
        shapes=shapes,
        adjacent_placement=adjacent_placement,
        projection=projection,
    )


_PROJECTORS: dict[tuple[str, str, str], tuple[SectionProjector | None, str]] = {}
"""Projectors by (volume, registration root, corrections root), kept across inventory refreshes so the stos files are
parsed once per process."""

_WARNED: set[tuple[str, str]] = set()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def attach_projection(
    volume: str,
    shapes: VolumeShapes,
    *,
    registration_root: Path | str | None,
    corrections_root: Path | str | None,
    residual: str = RESIDUAL_AUTO,
) -> dict[str, Any]:
    """Project ``shapes``' neighbour outlines (sets ``shapes.projected``) and return the record of how.

    Without the data the volume keeps the centroid placement: ``shapes.projected`` stays None, a warning is printed
    once per volume and reason, and the record says why.
    """
    cache_key = (volume, str(registration_root), str(corrections_root))
    if cache_key not in _PROJECTORS:
        _PROJECTORS[cache_key] = projector_for_volume(
            volume,
            registration_root=None if registration_root is None else Path(registration_root),
            corrections_root=None if corrections_root is None else Path(corrections_root),
        )
    projector, note = _PROJECTORS[cache_key]
    if projector is None:
        if (volume, note) not in _WARNED:
            _WARNED.add((volume, note))
            print(f"WARNING: adjacent outlines of {volume} are not projected ({note}); using the centroid placement")
        return {"placement": PLACEMENT_CENTROID, "reason": note}
    outlines, stats = project_volume_shapes(shapes, projector, residual)
    projector.stos.close()  # the transforms stay parsed; do not hold the zip open on the share
    shapes.projected = outlines
    record: dict[str, Any] = {
        "placement": PLACEMENT_PROJECTED,
        "registration": {
            "vikingxml": str(projector.stos.source),
            "stos_group": projector.stos.name,
            "stos_count": len(projector.stos.entries),
            "reference_sections": sorted(projector.stos.reference_sections),
            "nm_per_px": projector.nm_per_px,
        },
        "corrections": None,
        **stats,
    }
    if projector.corrections is not None:
        corrections = projector.corrections
        record["corrections"] = {
            "directory": str(corrections.directory),
            "stos_group": corrections.stos_group,
            "volume_url": corrections.manifest.get("volume_url"),
            "provenance": corrections.provenance,
            "manifest_sha256": _file_sha256(corrections.directory / "manifest.json"),
            "sections": len(corrections.manifest.get("z") or ()),
        }
    carry = ""
    if stats.get("carry_median_px_stos") is not None and stats.get("carry_median_px_residual") is not None:
        carry = (
            f" (centroid carry median {stats['carry_median_px_stos']:.0f} px stos only, "
            f"{stats['carry_median_px_residual']:.0f} px with the residual)"
        )
    print(
        f"adjacent outlines {volume}: {stats['projected']} of {stats['links']} links projected "
        f"({stats['failed']} outside a stos grid); residual {stats['residual']}{carry}; {note}"
    )
    return record


def adjacent_placement_summary(context: TileContext, volumes: Sequence[str]) -> str:
    """One line for the start-up banner: where ``adjacent_poly_seed`` outlines come from in each of ``volumes``.

    ``adjacent_poly_seed: projected for [RPC1, RPC2] / centroid fallback for [RC1, RC2] / no shapes.jsonl for [...]``.
    A centroid fallback (no registration data) makes adjacent seeds rare, so the line says what to set.
    """
    wanted = sorted({str(v) for v in volumes})
    no_shapes = [v for v in wanted if v not in context.shapes]
    fallback: list[str] = []
    if context.adjacent_placement != PLACEMENT_PROJECTED:
        placed = [v for v in wanted if v in context.shapes]
        parts = [f"{context.adjacent_placement} placement (configured) for [{', '.join(placed)}]"]
    else:
        projected = [v for v in wanted if context.projection.get(v, {}).get("placement") == PLACEMENT_PROJECTED]
        fallback = [v for v in wanted if v in context.shapes and v not in projected]
        parts = [f"projected for [{', '.join(projected)}]", f"centroid fallback for [{', '.join(fallback)}]"]
    if no_shapes:
        parts.append(f"no {SHAPES_FILENAME} for [{', '.join(no_shapes)}]")
    line = "adjacent_poly_seed: " + " / ".join(parts)
    if fallback:
        reasons = sorted({str(context.projection.get(v, {}).get("reason", "not projected")) for v in fallback})
        line += (
            f" ({'; '.join(reasons)}; projecting needs BOTH data.registration_root and data.corrections_root, "
            "otherwise adjacent seeds are rare)"
        )
    return line
