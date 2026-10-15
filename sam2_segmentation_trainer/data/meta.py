"""One training sample's labels. The string form is ``{volume}/{image_key}#{location_id}``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def format_sample_label(volume: object, image_key: object, location_id: object) -> str:
    """Return ``{volume}/{image_key}#{location_id}``."""
    return f"{volume}/{image_key}#{location_id}"


@dataclass(frozen=True, slots=True)
class SampleMeta:
    """Fields collate, MQTT labels, and location-score rows share."""

    volume: str
    image_key: str
    location_id: int
    downsample: int
    category_name: str
    split_key: str
    full_tile: bool = False
    """The target fills at least ``data.full_tile_threshold`` of the tile; validation scores these apart."""
    approved: bool = False
    """The mask is in ``approved.json``; validation reports these as a reference set."""
    has_negatives: bool = False
    """The tile holds other annotations (negative context) for this sample; validation splits circle-seed results by it."""

    def as_dict(self) -> dict[str, Any]:
        """Mapping stored on a collated batch. Keys match this record."""
        return {
            "volume": self.volume,
            "image_key": self.image_key,
            "location_id": self.location_id,
            "downsample": self.downsample,
            "category_name": self.category_name,
            "split_key": self.split_key,
            "full_tile": self.full_tile,
            "approved": self.approved,
            "has_negatives": self.has_negatives,
        }

    @property
    def label(self) -> str:
        return format_sample_label(self.volume, self.image_key, self.location_id)
