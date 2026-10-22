"""gold/raster.py: even-odd fill by pixel centre, exact on pixel-edge coordinates."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gold.raster import fill_rings  # noqa: E402


def test_a_pixel_aligned_rectangle_fills_exactly_its_area() -> None:
    mask = fill_rings([[(4, 4), (12, 4), (12, 10), (4, 10), (4, 4)]], 16, 16)
    assert int(mask.sum()) == 48 and mask[4:10, 4:12].all() and not mask[:4].any() and not mask[:, 12:].any()


def test_a_half_pixel_edge_includes_a_pixel_only_when_its_centre_is_inside() -> None:
    mask = fill_rings([[(0.4, 0), (3.6, 0), (3.6, 1), (0.4, 1)]], 8, 2)
    assert mask[0].tolist() == [True, True, True, True, False, False, False, False]


def test_a_hole_is_even_odd() -> None:
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    mask = fill_rings([outer, hole], 10, 10)
    assert int(mask.sum()) == 100 - 16 and not mask[4, 4] and mask[1, 1]


def test_a_circle_has_about_pi_r_squared_pixels_and_orientation_does_not_matter() -> None:
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    circle = [(50 + 30 * np.cos(a), 50 + 30 * np.sin(a)) for a in angles]
    a = fill_rings([circle], 100, 100)
    b = fill_rings([circle[::-1]], 100, 100)
    assert np.array_equal(a, b) and abs(int(a.sum()) - np.pi * 900) < 40


def test_clipping_and_degenerate_input() -> None:
    assert not fill_rings([], 8, 8).any()
    assert not fill_rings([[(1, 1), (2, 2)]], 8, 8).any()
    clipped = fill_rings([[(-5, -5), (20, -5), (20, 20), (-5, 20)]], 8, 8)
    assert clipped.all()


def test_a_large_ring_is_fast() -> None:
    angles = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    ring = [(512 + 400 * np.cos(a) * (1 + 0.05 * np.sin(9 * a)), 512 + 400 * np.sin(a)) for a in angles]
    start = time.time()
    mask = fill_rings([ring], 1024, 1024)
    assert time.time() - start < 2.0 and mask.sum() > 400000
