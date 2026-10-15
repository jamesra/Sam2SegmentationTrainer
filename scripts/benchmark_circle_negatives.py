"""Time the circle-negative encoding of edge-confidence-v5 against v6 (needs numpy and scipy only).

v5 rasterized the shrunk circle and ran a distance transform over it (``rasterize_circle`` + ``negative_canvas``).
v6 evaluates the radius-relative trust analytically inside the circle's bounding box
(``circle_negative_canvas``). Both build the same kind of object, a claimed mask and a ``-SEED_LOGIT`` canvas.

    python scripts/benchmark_circle_negatives.py [--tile 1024] [--repeat 50]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from annotation_crops.negatives import rasterize_circle
from annotation_crops.prompt_encoding import (
    DEFAULT_CIRCLE_TRUST,
    SEED_LOGIT,
    circle_negative_canvas,
    negative_canvas,
)


def _best_ms(function, repeat: int) -> float:
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--radii", default="20,60,150,400", help="True radii in pixels")
    args = parser.parse_args(argv)
    shape = (args.tile, args.tile)
    centre = args.tile / 2.0
    print(f"tile {args.tile}x{args.tile}, median of {args.repeat} runs, one circle per call")
    print(f"{'true r':>8} {'v5 ms':>9} {'v6 ms':>9} {'speed-up':>9} {'claimed px':>11}")
    for radius in (float(r) for r in args.radii.split(",")):
        def v5() -> None:
            mask = rasterize_circle(shape, centre, centre, radius * DEFAULT_CIRCLE_TRUST.zero_frac)
            negative_canvas(mask, seed_logit=SEED_LOGIT)

        def v6() -> None:
            circle_negative_canvas(shape, centre, centre, radius, DEFAULT_CIRCLE_TRUST, SEED_LOGIT)

        claimed = int(circle_negative_canvas(shape, centre, centre, radius).mask.sum())
        slow, fast = _best_ms(v5, args.repeat), _best_ms(v6, args.repeat)
        print(f"{radius:8.0f} {slow:9.2f} {fast:9.2f} {slow / fast:8.1f}x {claimed:11d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
