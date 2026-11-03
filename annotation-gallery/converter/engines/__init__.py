"""The two ways a location is segmented, kept apart so neither can pick up the other's rules.

* **Tiled** (:mod:`converter.engines.tiled`): the model was trained on tiles at one downsample (DS 1 for every model so
  far). The request is made at that level from the tiles under the prompt, and the server grows the object across as
  many 1024 tiles as it needs. A circle is its shape alone; a polygon is its shape and one click at its deepest point.
  This is how Viking's tile grid segments, and how the trainer saw the data.
* **Single window** (:mod:`converter.engines.window`): a model trained on whole objects fitted into one 1024 image.
  The request is made at the finest level where the annotation (and then the proposal) fits one window, with the
  seeds of :mod:`converter.seed`.

The served model says which it is: ``submission_mode`` in its manifest's client contract (``ModelInfo.contract_json``),
``tile_grid`` unless it says ``single_window``, and ``tile_downsample`` for the tiled level (1 unless stated). The card
the gallery draws is separate from either: it is laid out afterwards at whatever level holds the proposal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from converter import contour as contour_module
from converter.seg_client import SegmentResult

LOGGER = logging.getLogger("converter.engines")

TILED = "tile_grid"
WINDOW = "single_window"
DEFAULT_TILE_DOWNSAMPLE = 1


@dataclass
class Segmentation:
    """One location's proposal from an engine."""

    # The proposed outline, in pixels of ``level`` (mosaic pixels / level), Y up.
    shape: contour_module.Contour
    level: int
    result: SegmentResult
    # The card level the engine already settled on (the window engine segments at its card's level); None leaves the
    # card's layout to the caller.
    card_level: Optional[int] = None
    # Mosaic bounds the card at ``card_level`` must hold (None with no card level).
    card_bounds: Optional[tuple] = None


@dataclass(frozen=True)
class EngineChoice:
    kind: str
    tile_downsample: int = DEFAULT_TILE_DOWNSAMPLE


def contract(client: Any) -> dict:
    """The served model's client contract, or {} when the server cannot say (then the defaults apply)."""
    try:
        text = client.status().model.contract_json
    except Exception as error:  # noqa: BLE001 - a missing status means the defaults, not a failed build
        LOGGER.info("model contract not read (%s); using the tiled engine at DS %d", error, DEFAULT_TILE_DOWNSAMPLE)
        return {}
    try:
        parsed = json.loads(text) if text else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def choose(client: Any, forced: Optional[str] = None) -> EngineChoice:
    """Which engine the served model needs. ``forced`` (tests, a deliberate override) wins."""
    terms = contract(client)
    kind = forced or (WINDOW if str(terms.get("submission_mode") or "") == WINDOW else TILED)
    if kind not in (TILED, WINDOW):
        raise ValueError(f"unknown segmentation engine {kind!r}")
    try:
        level = int(terms.get("tile_downsample") or DEFAULT_TILE_DOWNSAMPLE)
    except (TypeError, ValueError):
        level = DEFAULT_TILE_DOWNSAMPLE
    return EngineChoice(kind=kind, tile_downsample=max(1, level))
