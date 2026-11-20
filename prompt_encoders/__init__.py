"""Mask-prompt encodings and display mapping shared by the SAM2 trainer and the segmentation server.

One module per prompt encoding, named by the encoding's version, so a served model is encoded exactly as it was
trained (``serve-train-prompt-parity``). Both sides import this package; there are no copies:

* :mod:`prompt_encoders.v6`: ``edge-confidence-v6``, the encoding models v4 and v5 trained with (formerly
  ``annotation_crops/prompt_encoding.py`` and the server's ``trainer_prompt_encoding.py``).
* :mod:`prompt_encoders.crush_display`: the gray Viking displays for a crushed (quantized) tileset (formerly
  ``annotation_crops/crush_display.py`` and the server's ``crush_display.py``).

``legacy-flat-v1`` has no module: the server keeps that path as it was. ``edge-confidence-v7`` (depth-relative trust)
is still built inside the server (``depth_trust.py``) and moves here with phase 3 of the move plan.
"""

from __future__ import annotations

from types import ModuleType
from typing import Dict, Tuple

from prompt_encoders import v6

LEGACY_ENCODING = "legacy-flat-v1"

_BY_VERSION: Dict[str, ModuleType] = {v6.PROMPT_ENCODING_VERSION: v6}


def versions() -> Tuple[str, ...]:
    """The encoding versions this package implements."""
    return tuple(_BY_VERSION)


def for_version(name: str) -> ModuleType:
    """The encoding module for version ``name``; ``KeyError`` naming the known versions when there is none."""
    try:
        return _BY_VERSION[name]
    except KeyError:
        raise KeyError(f"no prompt encoding {name!r} in prompt_encoders (have {', '.join(_BY_VERSION)})") from None


__all__ = ["LEGACY_ENCODING", "for_version", "versions", "v6"]
