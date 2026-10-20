"""Property tests for the geometry Update All leans on: tile coverage, tile-file parsing, circle shapes and windows.

Every property compares the code with a slow reference that is obviously right (a lattice walk, exact integer
arithmetic, a known circle), over coordinates the size of a real volume (hundreds of thousands of pixels).
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from circle_update import (  # noqa: E402
    Circle,
    CropWindowRef,
    annotated_circle,
    circle_geometry_from_wkt,
    entries_for_circle,
)
from refresh import TileIndex, covers_box  # noqa: E402

FAST = settings(max_examples=60, deadline=None)

# Real crop pyramids use power-of-two downsamples, so scaled coordinates stay exact in floating point.
DOWNSAMPLES = st.sampled_from([1, 2, 4, 8, 16, 32, 64])


# ------------------------------------------------------------------------------------------------ covers_box

# Rectangles on a lattice: origin plus step times a small index. Everything is an exact integer, so the
# reference can test every lattice cell without rounding.
_LATTICE = st.tuples(st.integers(-500_000, 500_000), st.integers(1, 4096))
_CELLS = 7


def _lattice_rect(draw, cells: int = _CELLS) -> tuple[int, int, int, int]:
    a = draw(st.integers(0, cells - 1))
    b = draw(st.integers(a + 1, cells))
    c = draw(st.integers(0, cells - 1))
    d = draw(st.integers(c + 1, cells))
    return a, b, c, d


@st.composite
def _coverage_case(draw):
    (x_origin, x_step), (y_origin, y_step) = draw(_LATTICE), draw(_LATTICE)
    box = _lattice_rect(draw)
    rects = [_lattice_rect(draw) for _ in range(draw(st.integers(0, 7)))]
    return (x_origin, x_step, y_origin, y_step), box, rects


def _brute_force_cover(box, rects) -> bool:
    """Walk every unit lattice cell of the box and ask whether some rectangle contains the cell."""
    for col in range(box[0], box[1]):
        for row in range(box[2], box[3]):
            if not any(r[0] <= col and col + 1 <= r[1] and r[2] <= row and row + 1 <= r[3] for r in rects):
                return False
    return True


def _scaled(rect, frame) -> tuple[float, float, float, float]:
    x_origin, x_step, y_origin, y_step = frame
    return (
        float(x_origin + rect[0] * x_step),
        float(y_origin + rect[2] * y_step),
        float(x_origin + rect[1] * x_step),
        float(y_origin + rect[3] * y_step),
    )


@FAST
@given(_coverage_case())
def test_covers_box_matches_a_cell_by_cell_walk(case) -> None:
    frame, box, rects = case
    expected = _brute_force_cover(box, rects)
    assert covers_box(_scaled(box, frame), [_scaled(r, frame) for r in rects]) is expected


@FAST
@given(_coverage_case(), st.randoms())
def test_covers_box_ignores_order_and_extra_rectangles_never_uncover(case, random) -> None:
    frame, box, rects = case
    box_f, rects_f = _scaled(box, frame), [_scaled(r, frame) for r in rects]
    result = covers_box(box_f, rects_f)
    shuffled = list(rects_f)
    random.shuffle(shuffled)
    assert covers_box(box_f, shuffled) is result
    extra = _scaled((0, _CELLS, 0, _CELLS), frame)
    assert covers_box(box_f, rects_f + [extra]) is True
    if result:
        assert covers_box(box_f, rects_f + rects_f[:1]) is True


@FAST
@given(_LATTICE, _LATTICE, st.integers(1, 3), st.integers(1, 3))
def test_covers_box_two_tiles_with_a_one_cell_gap_do_not_cover(x_axis, y_axis, left_cells, right_cells) -> None:
    """Two tiles that leave a seam between them miss part of the box, even though their union spans it."""
    (x_origin, x_step), (y_origin, y_step) = x_axis, y_axis
    left = (x_origin, y_origin, x_origin + left_cells * x_step, y_origin + y_step)
    gap_end = x_origin + (left_cells + 1) * x_step
    right = (gap_end, y_origin, gap_end + right_cells * x_step, y_origin + y_step)
    box = (x_origin, y_origin, right[2], y_origin + y_step)
    assert covers_box(tuple(map(float, box)), [tuple(map(float, left)), tuple(map(float, right))]) is False
    touching = (left[2], y_origin, gap_end, y_origin + y_step)
    assert covers_box(tuple(map(float, box)), [tuple(map(float, r)) for r in (left, touching, right)]) is True


@FAST
@given(
    st.floats(-1e6, 1e6, allow_nan=False),
    st.floats(-1e6, 1e6, allow_nan=False),
    st.floats(1e-3, 1e5, allow_nan=False),
    st.floats(1e-3, 1e5, allow_nan=False),
)
def test_covers_box_one_rectangle_containing_the_box_covers_it(x, y, width, height) -> None:
    box = (x, y, x + width, y + height)
    assert covers_box(box, [box]) is True
    assert covers_box(box, [(x - 1.0, y - 1.0, x + width + 1.0, y + height + 1.0)]) is True
    assert covers_box(box, []) is False


# ------------------------------------------------------------------------------------------------- TileIndex

_SUFFIXES = st.sampled_from([".png", ".jpg", ".jpeg", ".PNG", ".JPG"])
# Zero and negative extents are rare in a uniform range, so list them: the index must reject an empty or inverted window.
_EXTENT = st.one_of(st.integers(-3, 3), st.integers(1, 4096))
_PREFIXES = st.sampled_from(["RC2", "TEM_vol", "a_1_b", "R2_3", "x"])


@st.composite
def _tile_names(draw):
    """File names in an ``images`` folder: valid keys, keys the index must reject, and files it must ignore."""
    count = draw(st.integers(0, 8))
    entries = []
    seen = set()
    for _ in range(count):
        prefix, z = draw(_PREFIXES), draw(st.integers(0, 5000))
        downsample = draw(st.integers(0, 64))
        x0 = draw(st.integers(-500_000, 500_000))
        x1 = x0 + draw(_EXTENT)
        y0 = draw(st.integers(-500_000, 500_000))
        y1 = y0 + draw(_EXTENT)
        stem = f"{prefix}_{z}_D{downsample}_X{x0}-{x1}_Y{y0}-{y1}"
        if stem.lower() in seen:
            continue
        seen.add(stem.lower())
        entries.append((stem, draw(_SUFFIXES), z, downsample, x0, x1, y0, y1))
    return entries


@FAST
@given(_tile_names(), st.lists(st.sampled_from(["notes.txt", "RC2_1_D1_X0-1_Y0-1.json", "RC2_1_D1_X0-1.png"]), max_size=3))
def test_tile_index_lists_exactly_the_valid_image_files_by_section(entries, junk) -> None:
    with tempfile.TemporaryDirectory() as folder:
        images = Path(folder) / "images"
        images.mkdir()
        for stem, suffix, *_ in entries:
            (images / f"{stem}{suffix}").write_bytes(b"")
        for name in junk:
            (images / name).write_bytes(b"")
        # A folder with a perfect tile name is not a tile image.
        (images / "RC2_9_D1_X0-8_Y0-8.png").mkdir()
        index = TileIndex(folder)

    expected: dict[int, set[tuple]] = {}
    for stem, _suffix, z, downsample, x0, x1, y0, y1 in entries:
        if downsample > 0 and x1 > x0 and y1 > y0:
            expected.setdefault(z, set()).add((stem, downsample, x0, x1, y0, y1))
    for z in {entry[2] for entry in entries} | {-1, 9}:
        got = {(w.key, w.downsample, w.x0, w.x1, w.y0, w.y1) for w in index.windows(z)}
        assert got == expected.get(z, set())
        assert len(index.windows(z)) == len(got)
    for z in expected:
        for window in index.windows(z):
            scale = window.downsample
            assert window.mosaic_rect() == (window.x0 * scale, window.y0 * scale, window.x1 * scale, window.y1 * scale)
            assert window.mosaic_rect()[2] > window.mosaic_rect()[0]
            assert window.mosaic_rect()[3] > window.mosaic_rect()[1]


def test_tile_index_without_an_images_folder_is_empty() -> None:
    with tempfile.TemporaryDirectory() as folder:
        assert TileIndex(folder).windows(17) == []


def test_tile_index_reads_negative_window_corners() -> None:
    with tempfile.TemporaryDirectory() as folder:
        images = Path(folder) / "images"
        images.mkdir()
        (images / "RC2_17_D4_X-512--128_Y-64-256.png").write_bytes(b"")
        (window,) = TileIndex(folder).windows(17)
    assert (window.downsample, window.x0, window.x1, window.y0, window.y1) == (4, -512, -128, -64, 256)


# ------------------------------------------------------------------------- circle_geometry_from_wkt

_CENTER = st.floats(-500_000, 500_000, allow_nan=False, allow_infinity=False)
_RADIUS = st.floats(2.0, 100_000, allow_nan=False, allow_infinity=False)


def _on_circle(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def _curve_polygon(points: list[tuple[float, float]]) -> str:
    body = ", ".join(f"{x!r} {y!r}" for x, y in points)
    return f"CURVEPOLYGON(CIRCULARSTRING({body}))"


def _center_tolerance(cx: float, cy: float, radius: float) -> float:
    """The circumcenter formula squares coordinates, so rounding grows with (|c| + r)^2 / r^2 times |c|."""
    magnitude = max(abs(cx), abs(cy)) + radius
    return 1e-15 * magnitude * (magnitude / radius) ** 2 + 1e-9


@FAST
@given(_CENTER, _CENTER, _RADIUS, st.floats(0, math.tau), st.floats(0.7, 2.0), st.floats(0.7, 2.0))
def test_three_points_on_a_circle_recover_its_center_and_radius(cx, cy, radius, start, first_gap, second_gap) -> None:
    angles = (start, start + first_gap, start + first_gap + second_gap)
    points = [_on_circle(cx, cy, radius, angle) for angle in angles]
    found = circle_geometry_from_wkt(_curve_polygon(points))
    assert found is not None
    tolerance = _center_tolerance(cx, cy, radius)
    assert abs(found[0] - cx) <= tolerance
    assert abs(found[1] - cy) <= tolerance
    assert abs(found[2] - radius) <= tolerance


@FAST
@given(_CENTER, _CENTER, _RADIUS, st.floats(0, math.tau))
def test_a_full_circle_written_as_two_antipodes_and_a_repeat_recovers_its_center_and_radius(cx, cy, radius, angle) -> None:
    """Viking writes a whole circle as its start point, the opposite point, and the start again."""
    first = _on_circle(cx, cy, radius, angle)
    second = _on_circle(cx, cy, radius, angle + math.pi)
    found = circle_geometry_from_wkt(_curve_polygon([first, second, first]).lower())
    assert found is not None
    tolerance = 1e-9 * (max(abs(cx), abs(cy)) + radius)
    assert abs(found[0] - cx) <= tolerance
    assert abs(found[1] - cy) <= tolerance
    assert abs(found[2] - radius) <= tolerance


@FAST
@given(
    st.floats(-100, 100, allow_nan=False),
    st.floats(-100, 100, allow_nan=False),
    st.floats(0.005, 2.0),
    st.floats(0, math.tau),
    st.floats(0.7, 2.0),
    st.floats(0.7, 2.0),
)
def test_sub_pixel_circles_are_recovered_not_mistaken_for_a_straight_line(cx, cy, radius, start, first_gap, second_gap) -> None:
    angles = (start, start + first_gap, start + first_gap + second_gap)
    found = circle_geometry_from_wkt(_curve_polygon([_on_circle(cx, cy, radius, angle) for angle in angles]))
    assert found is not None
    tolerance = _center_tolerance(cx, cy, radius)
    assert abs(found[0] - cx) <= tolerance
    assert abs(found[1] - cy) <= tolerance
    assert abs(found[2] - radius) <= tolerance


def test_collinear_points_fall_back_to_the_circle_around_their_extent() -> None:
    assert circle_geometry_from_wkt(_curve_polygon([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])) == (5.0, 0.0, 5.0)
    assert circle_geometry_from_wkt(_curve_polygon([(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)])) == (3.0, 4.0, 5.0)


@FAST
@given(_CENTER, _CENTER, st.floats(1e-6, 1e6, allow_nan=False))
def test_circle_text_round_trips_exactly(cx, cy, radius) -> None:
    assert circle_geometry_from_wkt(f"CIRCLE(({cx!r} {cy!r}), {radius!r})") == (cx, cy, radius)


@FAST
@given(_CENTER, _CENTER, st.floats(-1e6, 0, allow_nan=False))
def test_circle_text_with_no_positive_radius_is_rejected(cx, cy, radius) -> None:
    assert circle_geometry_from_wkt(f"CIRCLE(({cx!r} {cy!r}), {radius!r})") is None


@FAST
@given(
    _CENTER,
    _CENTER,
    st.floats(-1000, 1000, allow_nan=False),
    st.floats(-1000, 1000, allow_nan=False),
    st.lists(st.floats(0.1, 1.0), min_size=2, max_size=2),
)
def test_collinear_points_give_a_finite_positive_circle(x, y, dx, dy, fractions) -> None:
    """Three points on one line have no circumcircle; the answer is still a usable circle, never a crash or NaN."""
    assume(math.hypot(dx, dy) > 1.0)
    points = [(x, y), (x + dx * fractions[0], y + dy * fractions[0]), (x + dx, y + dy)]
    found = circle_geometry_from_wkt(_curve_polygon(points))
    assert found is not None
    assert all(math.isfinite(value) for value in found)
    assert found[2] > 0


@FAST
@given(_CENTER, _CENTER)
def test_three_identical_points_are_not_a_circle(x, y) -> None:
    assert circle_geometry_from_wkt(_curve_polygon([(x, y)] * 3)) is None


def test_text_that_is_not_a_circle_is_rejected() -> None:
    for text in ("", "POINT(1 2)", "CURVEPOLYGON(CIRCULARSTRING(0 0, 1 1))", "CIRCLE((1 2))", "CIRCLE((1 2), 0)"):
        assert circle_geometry_from_wkt(text) is None


@FAST
@given(_CENTER, _CENTER, st.floats(0.5, 1e4), st.one_of(st.none(), st.floats(-5, 1e4, allow_nan=False)))
def test_a_positive_stored_radius_wins_over_the_shape(cx, cy, wkt_radius, stored) -> None:
    circle = annotated_circle(7, f"CIRCLE(({cx!r} {cy!r}), {wkt_radius!r})", stored, "mito")
    assert circle is not None
    expected = stored if stored is not None and stored > 0 else wkt_radius
    assert (circle.center_x, circle.center_y, circle.radius) == (cx, cy, expected)


# ------------------------------------------------------------------------------------- entries_for_circle


@st.composite
def _circle_and_windows(draw):
    """A circle and windows placed against its edges, to within one downsampled pixel of touching."""
    center_x, center_y = draw(st.integers(-500_000, 500_000)), draw(st.integers(-500_000, 500_000))
    radius = draw(st.one_of(st.integers(1, 40), st.integers(1, 20_000)))
    windows = []
    for position in range(draw(st.integers(1, 6))):
        downsample = draw(DOWNSAMPLES)
        width = draw(st.one_of(st.integers(1, 8), st.integers(1, 2048)))
        height = draw(st.one_of(st.integers(1, 8), st.integers(1, 2048)))
        nudge_x, nudge_y = draw(st.integers(-1, 1)), draw(st.integers(-1, 1))
        # Along each axis the window starts just past the circle's far edge, ends just short of its near edge,
        # or straddles the center; "just" is within one downsampled pixel, so tangent-ish cases come up often.
        origin_x = draw(
            st.sampled_from(
                [
                    -((-(center_x + radius)) // downsample) + nudge_x,
                    (center_x - radius) // downsample - width + nudge_x,
                    center_x // downsample - draw(st.integers(0, width)),
                ]
            )
        )
        origin_y = draw(
            st.sampled_from(
                [
                    -((-(center_y + radius)) // downsample) + nudge_y,
                    (center_y - radius) // downsample - height + nudge_y,
                    center_y // downsample - draw(st.integers(0, height)),
                ]
            )
        )
        windows.append(CropWindowRef(f"RC2_1_D{downsample}_{position}", downsample, origin_x, origin_y, width, height))
    circle = Circle(
        draw(st.integers(1, 10**9)),
        float(center_x),
        float(center_y),
        float(radius),
        draw(st.sampled_from(["mito", "nucleus", ""])),
        draw(st.booleans()),
        draw(st.sampled_from(["annotated", "inscribed"])),
    )
    return circle, windows


def _distance_squared_to_window(circle: Circle, window: CropWindowRef) -> int:
    """Exact squared distance, in mosaic pixels, from the circle's center to the window's rectangle."""
    scale = window.downsample
    left, right = window.origin_x * scale, (window.origin_x + window.width) * scale
    top, bottom = window.origin_y * scale, (window.origin_y + window.height) * scale
    dx = max(left - int(circle.center_x), 0, int(circle.center_x) - right)
    dy = max(top - int(circle.center_y), 0, int(circle.center_y) - bottom)
    return dx * dx + dy * dy


@FAST
@given(_circle_and_windows())
def test_entries_are_exactly_the_windows_the_circle_overlaps(case) -> None:
    circle, windows = case
    radius_squared = int(circle.radius) ** 2
    for window in windows:
        assume(_distance_squared_to_window(circle, window) != radius_squared)  # tangency is pinned by an example
    found = entries_for_circle(circle, windows)
    expected = {w.image_key for w in windows if _distance_squared_to_window(circle, w) < radius_squared}
    assert set(found) == expected
    for window in windows:
        if window.image_key not in found:
            continue
        entry = found[window.image_key]
        scale = window.downsample
        assert abs(entry["center_x"] - (circle.center_x / scale - window.origin_x)) <= 0.0005 + 1e-9
        assert abs(entry["center_y"] - (circle.center_y / scale - window.origin_y)) <= 0.0005 + 1e-9
        assert abs(entry["radius"] - circle.radius / scale) <= 0.0005 + 1e-9
        assert entry["location_id"] == circle.location_id
        assert entry["structure_label"] == circle.label
        assert entry["has_polygon_mask"] is circle.has_polygon_mask
        assert entry["source"] == circle.source


def test_a_circle_that_only_touches_a_window_edge_does_not_reach_it() -> None:
    circle = Circle(1, 0.0, 0.0, 5.0, "", False, "annotated")
    touching = CropWindowRef("touch", 1, 3, 4, 10, 10)  # nearest corner is (3, 4), exactly 5 away
    overlapping = CropWindowRef("overlap", 1, 3, 3, 10, 10)  # nearest corner is (3, 3), about 4.24 away
    found = entries_for_circle(circle, [touching, overlapping])
    assert set(found) == {"overlap"}


def test_a_circle_centered_inside_a_window_reaches_it_even_when_tiny() -> None:
    circle = Circle(1, 500_000.25, 300_000.75, 0.001, "", True, "inscribed")
    window = CropWindowRef("inside", 8, 62_000, 37_000, 2048, 2048)
    entry = entries_for_circle(circle, [window])["inside"]
    assert entry["center_x"] == round(500_000.25 / 8 - 62_000, 3)
    assert entry["center_y"] == round(300_000.75 / 8 - 37_000, 3)
    assert entry["has_polygon_mask"] is True
