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

    def as_dict(self) -> dict[str, Any]:
        """Mapping stored on a collated batch. Keys match this record."""
        return {
            "volume": self.volume,
            "image_key": self.image_key,
            "location_id": self.location_id,
            "downsample": self.downsample,
            "category_name": self.category_name,
            "split_key": self.split_key,
        }

    @property
    def label(self) -> str:
        return format_sample_label(self.volume, self.image_key, self.location_id)
