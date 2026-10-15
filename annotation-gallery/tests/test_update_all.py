"""Update All: paging, the row walk, skips, the watermark, and adding a location to existing crops."""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from catalog import connect  # noqa: E402
from refresh import TileIndex, add_new_location, covers_box  # noqa: E402
from update_all import (  # noqa: E402
    Dependencies,
    initial_watermark,
    load_state,
    odata_literal,
    parse_odata_datetime,
    save_state,
    start_job,
)

KEY_A = "RC2_17_D1_X0-1024_Y0-1024"
KEY_B = "RC2_17_D1_X1024-2048_Y0-1024"


def _crops(tmp_path: Path, ids: tuple[int, ...] = (1, 2)) -> Path:
    crops = tmp_path / "AnnotationCrops"
    (crops / "masks").mkdir(parents=True)
    connection = connect(crops)
    for location_id in ids:
        connection.execute(
            "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, radius, "
            "image_key, image_relpath, mask_relpath, ignored) VALUES (?, 17, 7, 'old', 1, 'Cell', 5.0, ?, ?, ?, 0)",
            [location_id, KEY_A, f"images/{KEY_A}.png", f"masks/{KEY_A}_{location_id}.png"],
        )
    connection.commit()
    connection.close()
    return crops


class _FakeOData:
    """Serves the given entities in pages of `top`, with @odata.count and @odata.nextLink."""

    def __init__(self, entities: list[dict]) -> None:
        self.entities = entities
        self.requests: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                owner.requests.append(self.path)
                query = parse_qs(urlparse(self.path).query)
                top = int(query["$top"][0])
                skip = int(query.get("skip", ["0"])[0])
                chosen = owner.entities
                filter_text = query.get("$filter", [""])[0]
                if "ID eq" in filter_text:
                    wanted = {int(part.split("eq")[1]) for part in filter_text.split(" or ")}
                    chosen = [e for e in chosen if e["ID"] in wanted]
                page = chosen[skip:skip + top]
                body: dict = {"value": page, "@odata.count": len(chosen)}
                if skip + top < len(chosen):
                    body["@odata.nextLink"] = f"{self.path.split('&skip=')[0]}&skip={skip + top}"
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:  # noqa: D401
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _entity(location_id: int, modified: str, label: str = "new") -> dict:
    return {
        "ID": location_id, "Z": 17, "ParentID": 9, "LastModified": modified,
        "Parent": {"ID": 9, "TypeID": 3, "Label": label, "Type": {"ID": 3, "Name": "Cell"}},
    }


def _deps(odata: _FakeOData, *, skip_ids: set[int] = frozenset(), fail_ids: set[int] = frozenset()):
    calls = {"refresh": [], "add": [], "drop": []}

    def refresh(crops, entity, exporter):
        calls["refresh"].append(entity["ID"])
        if entity["ID"] in fail_ids:
            raise ValueError("boom")
        return [KEY_A]

    def add(crops, entity, index):
        calls["add"].append(entity["ID"])
        if entity["ID"] in skip_ids:
            return None, "tile image missing for part of it"
        record = types.SimpleNamespace(
            id=entity["ID"], z=17, parent_id=9, structure_label="new", type_id=3, type_name="Cell", radius=None
        )
        return {"record": record, "windows": [{"image_key": KEY_A, "area": 100}]}, None

    deps = Dependencies(
        get_json=__import__("update_all").http_get_json,
        refresh=refresh,
        add=add,
        drop_stale=lambda crops, keys: calls["drop"].append(list(keys)),
        tile_index=lambda crops: None,
    )
    return deps, calls


def _run(crops: Path, odata: _FakeOData, deps: Dependencies, page_size: int = 2):
    job, started = start_job(f"V{id(crops)}", crops, odata.url, "tiled", page_size=page_size, deps=deps)
    assert started
    job.join(20)
    return job.snapshot()


def test_time_helpers_round_trip() -> None:
    moment = parse_odata_datetime("2026-08-12T00:02:18.863333+00:00")
    assert odata_literal(moment) == "2026-08-12T00:02:18.863333Z"
    assert parse_odata_datetime("2026-08-12T00:02:18Z").tzinfo is not None
    assert parse_odata_datetime("garbage") is None


def test_initial_watermark_is_the_older_of_folder_and_oldest_mask(tmp_path: Path) -> None:
    crops = _crops(tmp_path)
    mask = crops / "masks" / "a_1.png"
    mask.write_bytes(b"x")
    old = 1_600_000_000
    os.utime(mask, (old, old))
    assert initial_watermark(crops).timestamp() <= old + 1


def test_walk_refreshes_adds_skips_and_ticks_per_page(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1, 2))
    save_state(crops, {"watermark": "2026-01-01T00:00:00.000000Z"})
    entities = [
        _entity(1, "2026-03-01T00:00:00Z"),
        _entity(2, "2026-03-02T00:00:00Z"),
        _entity(3, "2026-03-03T00:00:00Z"),   # new, added
        _entity(4, "2026-03-04T00:00:00Z"),   # new, tile missing -> skipped
        _entity(5, "2026-03-05T00:00:00Z"),   # new, added
    ]
    odata = _FakeOData(entities)
    try:
        deps, calls = _deps(odata, skip_ids={4})
        result = _run(crops, odata, deps, page_size=2)
    finally:
        odata.close()
    assert result["phase"] == "done"
    assert result["pages"] == 3 and result["page"] == 3          # one tick per OData page
    assert (result["refreshed"], result["added"], result["skipped"], result["failed"]) == (2, 2, 1, 0)
    assert calls["refresh"] == [1, 2] and calls["add"] == [3, 4, 5]
    assert len(calls["drop"]) == 3                                  # one drop per page
    assert result["skipLines"] == ["skip 4 z=17: tile image missing for part of it"]
    connection = connect(crops)
    ids = {row[0] for row in connection.execute("SELECT location_id FROM locations")}
    labels = {row[0]: row[1] for row in connection.execute("SELECT location_id, structure_label FROM locations")}
    connection.close()
    assert ids == {1, 2, 3, 5}                                      # 4 was not added
    assert labels[1] == "new"                                       # metadata refreshed from the Parent
    state = load_state(crops)
    assert state["watermark"] == "2026-03-05T00:00:00.000000Z"       # newest LastModified seen
    assert state["skipped_ids"] == [4]
    assert (crops / "update_all.log").read_text(encoding="utf-8").count("skip 4") == 1


def test_failure_keeps_the_watermark_and_skips_are_retried_next_run(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": "2026-01-01T00:00:00.000000Z", "skipped_ids": [4]})
    entities = [_entity(1, "2026-03-01T00:00:00Z"), _entity(4, "2026-02-01T00:00:00Z")]
    odata = _FakeOData(entities)
    try:
        deps, calls = _deps(odata, fail_ids={1})
        result = _run(crops, odata, deps, page_size=10)
    finally:
        odata.close()
    assert result["failed"] == 1 and result["phase"] == "failed"
    assert result["errors"][0].startswith("1: boom")
    state = load_state(crops)
    assert state["watermark"] == "2026-01-01T00:00:00.000000Z"       # not advanced after a failure
    assert 4 not in state["skipped_ids"]                              # the retry added it
    assert 4 in calls["add"]


def test_cancel_stops_without_advancing(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": "2026-01-01T00:00:00.000000Z"})
    odata = _FakeOData([_entity(1, "2026-03-01T00:00:00Z"), _entity(2, "2026-03-02T00:00:00Z")])
    gate = threading.Event()
    try:
        deps, _calls = _deps(odata)
        inner = deps.refresh

        def slow_refresh(crops_arg, entity, exporter):
            gate.wait(5)
            return inner(crops_arg, entity, exporter)

        deps.refresh = slow_refresh
        job, _started = start_job("cancel-test", crops, odata.url, "tiled", page_size=1, deps=deps)
        job.cancel()
        gate.set()
        job.join(20)
        result = job.snapshot()
    finally:
        odata.close()
    assert result["cancelled"] and result["phase"] == "cancelled"
    assert load_state(crops)["watermark"] == "2026-01-01T00:00:00.000000Z"


def test_covers_box() -> None:
    assert covers_box((10, 10, 90, 90), [(0, 0, 100, 100)])
    assert covers_box((10, 10, 150, 90), [(0, 0, 100, 100), (100, 0, 200, 100)])
    assert not covers_box((10, 10, 150, 90), [(0, 0, 100, 100)])
    assert not covers_box((10, 10, 150, 90), [(0, 0, 100, 100), (110, 0, 200, 100)])


def _fake_support(tmp_square: tuple[float, float, float, float]):
    x0, y0, x1, y1 = tmp_square

    def locate(entity, include_off_edge):
        return types.SimpleNamespace(
            id=entity["ID"], z=17, wkt="x", type_code=1, radius=None, parent_id=9,
            structure_label="L", type_id=3, type_name="Cell",
        )

    def make_job(output, location_id, key, polygons, **kw):
        return types.SimpleNamespace(
            image_key=key, location_id=location_id,
            mask_path=str(output / "masks" / f"{key}_{location_id}.png"),
            rle_path=str(output / "_work" / "rle" / f"{key}_{location_id}.json"),
        )

    def run(jobs, workers):
        out = []
        for job in jobs:
            Path(job.rle_path).parent.mkdir(parents=True, exist_ok=True)
            Path(job.rle_path).write_text(json.dumps({"counts": [1, 2], "size": [4, 4]}), encoding="utf-8")
            Path(job.mask_path).write_bytes(b"png")
            out.append({"image_key": job.image_key, "rle_path": job.rle_path, "bbox": [0, 0, 2, 2], "area": 4})
        return out

    return types.SimpleNamespace(
        location_from_odata_entity=locate,
        parse_wkt_polygons=lambda wkt: [((0, 0),)],
        hydrate_polygons=lambda polygons, code: polygons,
        mosaic_bbox=lambda polygons: (x0, y0, x1, y1),
        mask_job_for_crop=make_job,
        run_mask_jobs=run,
        sa1b_annotation=lambda record, **kw: {"id": record.id, **{k: v for k, v in kw.items() if k != "rle"}},
    )


def _tiled_crops(tmp_path: Path, keys: tuple[str, ...]) -> Path:
    crops = tmp_path / "AnnotationCrops"
    (crops / "images").mkdir(parents=True)
    (crops / "masks").mkdir()
    for key in keys:
        (crops / "images" / f"{key}.png").write_bytes(b"png")
        (crops / "images" / f"{key}.json").write_text(
            json.dumps({"image": {"width": 1024, "height": 1024}, "annotations": [{"id": 1}]}), encoding="utf-8"
        )
    return crops


def test_add_location_when_every_needed_tile_is_present(tmp_path: Path) -> None:
    crops = _tiled_crops(tmp_path, (KEY_A, KEY_B))
    index = TileIndex(crops)
    result, reason = add_new_location(crops, {"ID": 50}, index, _fake_support((900, 100, 1100, 300)))
    assert reason is None
    assert sorted(w["image_key"] for w in result["windows"]) == [KEY_A, KEY_B]
    for key in (KEY_A, KEY_B):
        payload = json.loads((crops / "images" / f"{key}.json").read_text(encoding="utf-8"))
        assert [a["id"] for a in payload["annotations"]] == [1, 50]
        assert payload["annotations"][1]["window_count"] == 2
    assert (crops / "masks" / f"{KEY_A}_50.png").is_file()


def test_add_location_is_skipped_whole_when_a_tile_is_missing(tmp_path: Path) -> None:
    crops = _tiled_crops(tmp_path, (KEY_A,))
    index = TileIndex(crops)
    result, reason = add_new_location(crops, {"ID": 50}, index, _fake_support((900, 100, 1100, 300)))
    assert result is None and reason == "tile image missing for part of it"
    assert not (crops / "masks" / f"{KEY_A}_50.png").exists()
    payload = json.loads((crops / "images" / f"{KEY_A}.json").read_text(encoding="utf-8"))
    assert [a["id"] for a in payload["annotations"]] == [1]
    other, other_reason = add_new_location(crops, {"ID": 51}, index, _fake_support((5000, 5000, 5100, 5100)))
    assert other is None and other_reason == "no tile image covers it"


# ------------------------------------------------------------------- circles

from circle_update import (  # noqa: E402
    Circle,
    CircleFile,
    CropWindowRef,
    annotated_circle,
    circle_geometry_from_wkt,
    entries_for_circle,
    inscribed_circle,
)


def test_circle_geometry_from_circularstring_and_circle_wkt() -> None:
    wkt = "CURVEPOLYGON(CIRCULARSTRING(90 100, 100 110, 110 100, 100 90, 90 100))"
    x, y, r = circle_geometry_from_wkt(wkt)
    assert (round(x, 6), round(y, 6), round(r, 6)) == (100.0, 100.0, 10.0)
    assert circle_geometry_from_wkt("CIRCLE((5 6), 7)") == (5.0, 6.0, 7.0)
    assert circle_geometry_from_wkt("POINT(1 2)") is None
    stored = annotated_circle(3, wkt, 12.5, "MC")
    assert stored.radius == 12.5 and stored.source == "annotated" and not stored.has_polygon_mask
    assert annotated_circle(3, wkt, None, "MC").radius == 10.0


def test_inscribed_circle_of_a_square_and_a_c_shape() -> None:
    square = [(((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),)]
    circle = inscribed_circle(9, square, "MC")
    assert abs(circle.center_x - 50) < 1 and abs(circle.center_y - 50) < 1 and abs(circle.radius - 50) < 1
    assert circle.source == "inscribed" and circle.has_polygon_mask
    c_shape = [(((0.0, 0.0), (100.0, 0.0), (100.0, 30.0), (30.0, 30.0), (30.0, 70.0),
                 (100.0, 70.0), (100.0, 100.0), (0.0, 100.0)),)]
    inside = inscribed_circle(9, c_shape, "")
    assert inside.radius > 13                               # the arms and the spine are each 30 wide
    x, y = inside.center_x, inside.center_y
    assert x <= 30 or y <= 30 or y >= 70                    # the center is in the C, not in its mouth


def test_entries_use_crop_pixels_including_negative_origins() -> None:
    circle = Circle(7, 1000.0, 200.0, 40.0, "MC", False, "annotated")
    windows = [
        CropWindowRef("RC2_17_D1_X1024-2048_Y0-1024", 1, 1024, 0, 1024, 1024),
        CropWindowRef("RC2_17_D1_X0-1024_Y0-1024", 1, 0, 0, 1024, 1024),
        CropWindowRef("RC2_17_D2_X-512-512_Y0-512", 2, -512, 0, 1024, 512),
    ]
    entries = entries_for_circle(circle, windows)
    assert set(entries) == {"RC2_17_D1_X1024-2048_Y0-1024", "RC2_17_D1_X0-1024_Y0-1024", "RC2_17_D2_X-512-512_Y0-512"}
    assert entries["RC2_17_D1_X1024-2048_Y0-1024"] == {
        "location_id": 7, "center_x": -24.0, "center_y": 200.0, "radius": 40.0,
        "structure_label": "MC", "has_polygon_mask": False, "source": "annotated",
    }
    assert entries["RC2_17_D1_X0-1024_Y0-1024"]["center_x"] == 1000.0
    d2 = entries["RC2_17_D2_X-512-512_Y0-512"]               # the window hangs over the mosaic edge
    assert d2["center_x"] == (1000.0 - (-512 * 2)) / 2 and d2["radius"] == 20.0
    far = entries_for_circle(Circle(7, 5000.0, 5000.0, 40.0, "", False, "annotated"), windows)
    assert far == {}


def test_circle_file_replace_keeps_other_entries_and_writes_once(tmp_path: Path) -> None:
    crops = tmp_path / "AnnotationCrops"
    crops.mkdir()
    (crops / "circle_annotations.json").write_text(json.dumps({
        "version": 1, "diameter_scale": 1.0,
        "grid_cells": {
            "A": [{"location_id": 1, "center_x": 1.0}, {"location_id": 2, "center_x": 2.0}],
            "B": [{"location_id": 2, "center_x": 3.0}],
        },
    }), encoding="utf-8")
    file = CircleFile(crops)
    assert file.exists
    file.replace(2, {"C": {"location_id": 2, "center_x": 9.0}})
    assert file.save()
    saved = json.loads((crops / "circle_annotations.json").read_text(encoding="utf-8"))
    assert saved["grid_cells"] == {"A": [{"location_id": 1, "center_x": 1.0}], "C": [{"location_id": 2, "center_x": 9.0}]}
    stamp = (crops / "circle_annotations.json").stat().st_mtime_ns
    assert CircleFile(crops).save() and (crops / "circle_annotations.json").stat().st_mtime_ns == stamp


def _circle_deps(odata: _FakeOData, circle_by_id: dict[int, Circle]):
    deps, calls = _deps(odata)
    deps.circle_for = lambda entity: circle_by_id.get(entity["ID"])
    deps.tile_index = lambda crops: types.SimpleNamespace(
        windows=lambda z: [types.SimpleNamespace(key=KEY_A, downsample=1, x0=0, y0=0, x1=1024, y1=1024)]
    )
    return deps, calls


def test_update_all_rewrites_circle_entries(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": "2026-01-01T00:00:00.000000Z"})
    (crops / "circle_annotations.json").write_text(json.dumps({
        "version": 1, "diameter_scale": 1.0,
        "grid_cells": {KEY_A: [
            {"location_id": 1, "center_x": 10.0, "center_y": 10.0, "radius": 5.0, "source": "inscribed"},
            {"location_id": 8, "center_x": 300.0, "center_y": 300.0, "radius": 20.0, "source": "annotated"},
        ]},
    }), encoding="utf-8")
    circle_entity = {**_entity(8, "2026-03-02T00:00:00Z"), "TypeCode": 1}
    odata = _FakeOData([_entity(1, "2026-03-01T00:00:00Z"), circle_entity])
    circles = {
        1: Circle(1, 500.0, 600.0, 55.0, "new", True, "inscribed"),     # polygon 1 was edited
        8: Circle(8, 700.0, 710.0, 33.0, "new", False, "annotated"),    # circle 8 moved and was resized
    }
    try:
        deps, calls = _circle_deps(odata, circles)
        result = _run(crops, odata, deps, page_size=10)
    finally:
        odata.close()
    assert result["phase"] == "done" and result["circles"] == 2
    assert calls["refresh"] == [1] and calls["add"] == []           # the circle location is never rasterized
    assert (result["refreshed"], result["skipped"]) == (1, 0)       # and is not counted as a skip
    entries = {e["location_id"]: e for e in json.loads(
        (crops / "circle_annotations.json").read_text(encoding="utf-8"))["grid_cells"][KEY_A]}
    assert entries[1]["center_x"] == 500.0 and entries[1]["radius"] == 55.0 and entries[1]["source"] == "inscribed"
    assert entries[8]["center_x"] == 700.0 and entries[8]["radius"] == 33.0 and entries[8]["source"] == "annotated"


def test_circles_are_not_created_when_the_volume_has_no_circle_file(tmp_path: Path) -> None:
    crops = _crops(tmp_path, ids=(1,))
    save_state(crops, {"watermark": "2026-01-01T00:00:00.000000Z"})
    odata = _FakeOData([{**_entity(8, "2026-03-02T00:00:00Z"), "TypeCode": 1}])
    try:
        deps, _calls = _circle_deps(odata, {8: Circle(8, 1.0, 1.0, 5.0, "", False, "annotated")})
        result = _run(crops, odata, deps, page_size=10)
    finally:
        odata.close()
    assert result["circles"] == 0 and any("circle_annotations.json is missing" in n for n in result["notes"])
    assert not (crops / "circle_annotations.json").exists()


def test_import_location_adds_missing_skips_untiled_and_ignores_known(tmp_path: Path) -> None:
    from update_all import import_location

    crops = _crops(tmp_path)
    deps, calls = _deps(_FakeOData([]), skip_ids={6})

    assert import_location(crops, _entity(1, "x"), deps=deps)["status"] == "exists"
    assert calls["add"] == []

    missing = import_location(crops, _entity(6, "x"), deps=deps)
    assert missing["status"] == "no_tiles" and "missing" in missing["reason"]
    assert 6 not in {row[0] for row in connect(crops).execute("SELECT location_id FROM locations")}

    done = import_location(crops, _entity(5, "x"), deps=deps)
    assert done == {"status": "imported", "location_id": 5, "imageKeys": [KEY_A]}
    assert calls["drop"] == [[KEY_A]]
    rows = connect(crops).execute("SELECT image_key FROM locations WHERE location_id = 5").fetchall()
    assert [row[0] for row in rows] == [KEY_A]
