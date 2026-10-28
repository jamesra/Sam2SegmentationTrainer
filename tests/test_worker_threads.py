"""DataLoader worker threading and the two output-identical data-path speed-ups (imagenet_normalize, erode_outline)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from sam2_segmentation_trainer.data.dataset import IMAGENET_MEAN, IMAGENET_STD, imagenet_normalize, limit_worker_threads
from sam2_segmentation_trainer.data.shapes import erode_outline


class WorkerThreadTests(unittest.TestCase):
    def test_every_loaded_openblas_is_set_to_one_thread_in_a_worker(self) -> None:
        # In a fresh process, so this test process keeps its own threading.
        script = textwrap.dedent(
            """
            import ctypes, os
            import numpy, scipy.ndimage, cv2, torch
            from sam2_segmentation_trainer.data import dataset as D
            paths = D._loaded_openblas_paths()
            D.limit_worker_threads(0)
            getters = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                       "openblas_get_num_threads64_", "openblas_get_num_threads")
            counts = []
            for path in paths:
                lib = ctypes.CDLL(path)
                getter = next(getattr(lib, n) for n in getters if hasattr(lib, n))
                counts.append(getter())
            import json
            print(json.dumps({"libs": len(paths), "blas": counts, "torch": torch.get_num_threads(),
                              "cv2": cv2.getNumThreads(), "omp": os.environ["OMP_NUM_THREADS"]}))
            """
        )
        if not sys.platform.startswith("linux"):
            self.skipTest("OpenBLAS libraries are found through /proc/self/maps")
        import json

        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
        state = json.loads(out.strip().splitlines()[-1])
        self.assertGreaterEqual(state["libs"], 1)  # numpy's OpenBLAS at least
        self.assertEqual(state["blas"], [1] * state["libs"])
        self.assertEqual((state["torch"], state["cv2"], state["omp"]), (1, 1, "1"))

    def test_training_and_validation_loaders_set_up_their_workers(self) -> None:
        from sam2_segmentation_trainer.train import _make_loader
        from sam2_segmentation_trainer.validation import ValidationSuite

        loader = _make_loader(list(range(4)), [0, 1], batch_size=1, num_workers=0, drop_last=True,
                              prefetch_factor=2, persistent_workers=False)
        self.assertIs(loader.worker_init_fn, limit_worker_threads)
        self.assertIn("worker_init_fn=limit_worker_threads", _source(ValidationSuite.loader))

    def test_the_main_process_keeps_its_threads(self) -> None:
        # Importing the trainer must not cap the main process (it drives the GPU and the metric threads).
        script = "import torch; before = torch.get_num_threads(); import sam2_segmentation_trainer.train; print(before, torch.get_num_threads())"
        before, after = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(before, after)


def _source(function) -> str:
    import inspect

    return inspect.getsource(function)


def _reference_normalize(rgb: np.ndarray) -> torch.Tensor:
    """The implementation before the fused one."""
    arr = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _reference_erode(canvas: np.ndarray, erode_px: float, max_share: float = 0.6) -> np.ndarray:
    """The implementation before the bounding-box one (the whole canvas transformed)."""
    if erode_px <= 0.0 or not canvas.any():
        return canvas
    depth = distance_transform_edt(canvas)
    return depth > min(float(erode_px), float(depth.max()) * max_share)


class IdenticalOutputTests(unittest.TestCase):
    def test_imagenet_normalize_is_bit_identical(self) -> None:
        rng = np.random.default_rng(0)
        for shape in ((1024, 1024, 3), (37, 53, 3)):
            rgb = rng.integers(0, 256, size=shape, dtype=np.uint8)
            new, old = imagenet_normalize(rgb), _reference_normalize(rgb)
            self.assertEqual(new.dtype, torch.float32)
            self.assertEqual(tuple(new.shape), (3, shape[0], shape[1]))
            self.assertTrue(torch.equal(new, old))
        every = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)  # every value in every channel
        self.assertTrue(torch.equal(imagenet_normalize(every), _reference_normalize(every)))

    def test_erode_outline_is_identical_on_the_bounding_box(self) -> None:
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:120, 0:140]
        checked = 0
        for _ in range(60):
            canvas = np.zeros((120, 140), dtype=bool)
            for _blob in range(int(rng.integers(1, 4))):
                cy, cx = rng.uniform(-10, 130), rng.uniform(-10, 150)  # some run off the canvas edge
                ry, rx = rng.uniform(5, 40), rng.uniform(5, 40)
                canvas |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            canvas[int(rng.integers(0, 120)), :] &= rng.random() < 0.5  # a cut through some of them
            erode = float(rng.uniform(0, 12))
            np.testing.assert_array_equal(erode_outline(canvas, erode), _reference_erode(canvas, erode))
            checked += int(canvas.any())
        self.assertGreater(checked, 40)
        full = np.ones((30, 30), dtype=bool)  # no background at all: the canvas edge is not an edge
        np.testing.assert_array_equal(erode_outline(full, 4.0), _reference_erode(full, 4.0))


if __name__ == "__main__":
    unittest.main()


class ForkSafetyTests(unittest.TestCase):
    """cv2.setNumThreads(1) inside a forked worker hangs when the parent already ran a parallel cv2 operation; the parent
    sets one thread first, so the worker's call is skipped."""

    def test_main_process_sets_one_cv2_thread_and_worker_skips_the_call(self) -> None:
        import cv2

        from sam2_segmentation_trainer.data.dataset import limit_main_process_cv2, limit_worker_threads

        before = cv2.getNumThreads()
        try:
            limit_main_process_cv2()
            self.assertEqual(cv2.getNumThreads(), 1)
            with mock.patch.object(cv2, "setNumThreads") as setter:
                limit_worker_threads(0)
            setter.assert_not_called()
        finally:
            cv2.setNumThreads(before)
