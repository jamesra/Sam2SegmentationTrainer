"""Locations of one structure from a volume's OData endpoint: the query, the paging, and the record.

Read-only. The converter never writes to OData (it cannot); approved results go to a batch file (see ``batch.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from converter import geo

SELECT = "ID,ParentID,TypeCode,Z,X,Y,OffEdge,MosaicShape,LastModified,Radius"
EXPAND = "Parent($select=ID,TypeID,Label;$expand=Type($select=ID,Name,ParentID))"
PAGE_SIZE = 500

GetJson = Callable[[str], Dict[str, Any]]

# Types the converter builds cards for by default: the closed shapes that become polygons, and circles.
DEFAULT_TYPE_CODES = (1, 4, 6, 7)


@dataclass(frozen=True)
class LocationRecord:
    """One location. ``rings`` are the mosaic control-point rings (exterior first); a circle has a centre and radius."""

    id: int
    structure_id: Optional[int]
    z: int
    type_code: int
    wkt: str
    radius: Optional[float]
    off_edge: bool
    last_modified: str
    structure_label: Optional[str] = None
    type_name: Optional[str] = None
    tags: Tuple[str, ...] = ()
    center: Optional[Tuple[float, float]] = None
    polygons: Tuple[geo.PolygonRings, ...] = field(default=())

    @property
    def is_circle(self) -> bool:
        return self.type_code == 1


def _tags(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return tuple(sorted(t for t in (part.strip() for part in value.replace(";", ",").split(",")) if t))
    if isinstance(value, list):
        return tuple(sorted(str(t) for t in value if t))
    return ()


def _center_of(entity: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """A circle's centre is the location's own X, Y (mosaic pixels). Its MosaicShape is a ring of rim points whose first
    point is the rightmost one, so the shape must not be used for it."""
    x, y = entity.get("X"), entity.get("Y")
    if x in (None, "") or y in (None, ""):
        return None
    return float(x), float(y)


def record_from_entity(entity: Dict[str, Any]) -> Optional[LocationRecord]:
    """A record for a seedable location, else None (points, open curves, ellipses, polylines, empty shapes)."""
    raw_type = entity.get("TypeCode")
    if raw_type is None or raw_type == "":
        return None
    type_code = int(raw_type)
    wkt = geo.mosaic_wkt_from_odata_shape(entity.get("MosaicShape"))
    if not wkt:
        return None
    parent = entity.get("Parent") or {}
    parent_type = parent.get("Type") or {}
    raw_radius = entity.get("Radius")
    radius = None if raw_radius in (None, "") else float(raw_radius)
    base = dict(
        id=int(entity["ID"]),
        structure_id=entity.get("ParentID", parent.get("ID")),
        z=int(entity["Z"]),
        type_code=type_code,
        wkt=wkt,
        radius=radius,
        off_edge=bool(entity.get("OffEdge", False)),
        last_modified=str(entity.get("LastModified") or ""),
        structure_label=parent.get("Label"),
        type_name=parent_type.get("Name"),
        tags=_tags(entity.get("Tags")),
    )
    if base["structure_id"] is not None:
        base["structure_id"] = int(base["structure_id"])
    if type_code == 1:
        center = _center_of(entity)
        if center is None or not radius or radius <= 0:
            return None
        return LocationRecord(center=center, **base)
    if type_code not in (4, 6, 7):
        return None
    try:
        polygons = geo.parse_wkt_polygons(wkt)
    except ValueError:
        return None
    if not polygons:
        return None
    return LocationRecord(polygons=tuple(geo.hydrate_polygons(polygons, type_code)), **base)


def bounds(record: LocationRecord) -> Tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) in mosaic pixels."""
    if record.is_circle:
        (cx, cy), r = record.center, record.radius
        return cx - r, cy - r, cx + r, cy + r
    xs = [x for rings in record.polygons for ring in rings[:1] for x, _ in ring]
    ys = [y for rings in record.polygons for ring in rings[:1] for _, y in ring]
    return min(xs), min(ys), max(xs), max(ys)


def filter_text(
    structure_id: int,
    *,
    type_codes: Sequence[int] = DEFAULT_TYPE_CODES,
    z_min: Optional[int] = None,
    z_max: Optional[int] = None,
) -> str:
    """``$filter`` for one structure's locations: the types and Z range the user chose. Off-edge locations are included."""
    clauses = [f"ParentID eq {int(structure_id)}"]
    if type_codes:
        clauses.append("(" + " or ".join(f"TypeCode eq {int(code)}" for code in type_codes) + ")")
    if z_min is not None:
        clauses.append(f"Z ge {int(z_min)}")
    if z_max is not None:
        clauses.append(f"Z le {int(z_max)}")
    return " and ".join(clauses)


def locations_url(base: str, filter_expression: str) -> str:
    """The locations matching the filter in ID order. No ``$top``: it caps the whole answer, not the page size."""
    query = (
        f"$filter={quote(filter_expression, safe='()/')}"
        f"&$orderby=ID&$select={quote(SELECT, safe=',')}"
        f"&$expand={quote(EXPAND, safe='();$,')}"
        f"&$count=true"
    )
    return f"{base.rstrip('/')}/Locations?{query}"


def http_get_json(url: str, *, max_page_size: Optional[int] = PAGE_SIZE, timeout: float = 120.0) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    if max_page_size:
        headers["Prefer"] = f"odata.maxpagesize={int(max_page_size)}"
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OData response is not an object")
    return payload


def iter_pages(url: str, get_json: GetJson = http_get_json) -> Iterator[Dict[str, Any]]:
    """Every page of an answer: follow ``@odata.nextLink`` until a page leaves it out."""
    next_url: Optional[str] = url
    while next_url:
        document = get_json(next_url)
        yield document
        link = document.get("@odata.nextLink")
        if not link:
            return
        next_url = str(link) if urlparse(str(link)).scheme else urljoin(next_url, str(link))


def iter_locations(
    base: str,
    structure_id: int,
    *,
    get_json: GetJson = http_get_json,
    **filters: Any,
) -> Iterator[LocationRecord]:
    """The structure's seedable locations, page by page."""
    url = locations_url(base, filter_text(structure_id, **filters))
    for page in iter_pages(url, get_json):
        for entity in page.get("value") or []:
            if isinstance(entity, dict) and "ID" in entity:
                record = record_from_entity(entity)
                if record is not None:
                    yield record


def links_url(base: str, structure_id: int) -> str:
    """Every location link with an end in the structure's locations, as ``A`` and ``B`` location ids."""
    return f"{base.rstrip('/')}/StructureLocationLinks(StructureID={int(structure_id)})?$select=A,B&$orderby=A,B"


def iter_links(base: str, structure_id: int, *, get_json: GetJson = http_get_json) -> Iterator[Tuple[int, int]]:
    """The structure's location links page by page, each as ``(a, b)`` with ``a < b``; a link is stored once."""
    for page in iter_pages(links_url(base, structure_id), get_json):
        for entity in page.get("value") or []:
            if isinstance(entity, dict) and entity.get("A") is not None and entity.get("B") is not None:
                a, b = int(entity["A"]), int(entity["B"])
                if a != b:
                    yield (a, b) if a < b else (b, a)


def count_locations(base: str, structure_id: int, *, get_json: GetJson = http_get_json, **filters: Any) -> int:
    """How many locations match, from ``$count`` on a one-row page."""
    url = f"{base.rstrip('/')}/Locations?$filter={quote(filter_text(structure_id, **filters), safe='()/')}&$count=true&$top=1&$select=ID"
    document = get_json(url)
    value = document.get("@odata.count")
    if value is None:
        raise ValueError("the endpoint did not return @odata.count")
    return int(value)
