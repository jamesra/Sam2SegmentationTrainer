"""A stand-in for Viking's native prompt library in tests that run without it (the Windows dev checkout).

It shapes requests as the library does where the tests look (grid naming, circle disc and centre click, polygon as an
outline to correct, location refs) and records what it was given. The library's own rules are tested in Viking
(``SegmentationPrompts/Viking.SegmentationPrompts.Tests``) and checked in the gallery image (``check_native.py``).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from converter import geo
from seg_proto import prompt_input_pb2 as prompt_pb
from seg_proto import segmentation_pb2 as pb
from seg_proto.Common import geometry_pb2 as geometry


class FakePromptLibrary:
    def __init__(self, reach: Tuple[float, float, float, float] = (-1e9, -1e9, 1e9, 1e9)) -> None:
        self.reach_rect = reach
        self.inputs: List[prompt_pb.PromptInput] = []
        self.background_calls: List[Tuple[int, List[int]]] = []

    def build_start(self, prompt_input: prompt_pb.PromptInput) -> pb.SegmentTilesRequest:
        self.inputs.append(prompt_input)
        start = pb.SegmentTilesRequest(
            grid=geometry.TileGrid(
                volume=prompt_input.volume, section=prompt_input.section, channel=prompt_input.tileset,
                transform=prompt_input.tileset, downsample=1, space=geometry.COORDINATE_SPACE_MOSAIC,
            ),
            request_id=prompt_input.request_id, annotation_context_id=prompt_input.annotation_context_id,
        )
        for target in sorted(prompt_input.targets, key=lambda t: t.id):
            if target.type_code == 1:
                start.foreground_feature.shapes.add().circle.CopyFrom(
                    geometry.Circle(center=geometry.Vector2(x=target.x, y=target.y), radius=target.radius)
                )
                start.foreground_feature.points.add(x=target.x, y=target.y)
            else:
                for rings in geo.parse_wkt_polygons(target.mosaic_wkt):
                    polygon = start.correction_feature.shapes.add().polygon
                    polygon.exterior.points.extend(geometry.Vector2(x=x, y=y) for x, y in rings[0][:-1])
            start.locations.add(id=target.id, last_modified_ticks=target.last_modified_ticks)
        return start

    def reach(self, prompt_input: prompt_pb.PromptInput) -> Tuple[float, float, float, float]:
        return self.reach_rect

    def background_for(self, prompt_input, needed, sent_ids: Sequence[int]) -> prompt_pb.PromptBackground:
        self.background_calls.append((len(needed.background_cells), list(sent_ids)))
        return prompt_pb.PromptBackground()
