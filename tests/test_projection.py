"""Section-to-section projection of adjacent outlines: stos grids, residual fields, the overlap rule and the fallback."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sam2_segmentation_trainer.data.dataset import EMSegDataset
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.projection import (
    CONVENTION_NONE,
    GridStos,
    IdentityStos,
    SectionField,
    find_correction_set,
    load_stos_group,
    project_volume_shapes,
    projector_for_volume,
    volume_scale_nm,
)
from sam2_segmentation_trainer.data.prompts import AdjacentPoly, PromptConfig, PromptType, build_prompt
from sam2_segmentation_trainer.data.shapes import (
    SHAPES_FILENAME,
    AdjacentDraw,
    draw_projected,
    erode_outline,
    load_volume_shapes,
)
from sam2_segmentation_trainer.data.tiles import TileContext, attach_projection, build_tile_context
from sam2_segmentation_trainer.data_record import build_data_record

try:
    from tests.helpers import make_crops
    from tests.test_shapes import _record, _write
except ImportError:
    from helpers import make_crops
    from test_shapes import _record, _write

GROUP = "SliceToVolume1"


def stos_text(translation: tuple[float, float], width: int = 400, height: int = 300, grid=(4, 3)) -> str:
    """A grid stos whose control points are the mapped grid moved by ``translation`` (a pure translation)."""
    grid_w, grid_h = grid
    xs = np.arange(grid_w) / (grid_w - 1) * width
    ys = np.arange(grid_h) / (grid_h - 1) * height
    values = []
    for y in ys:
        for x in xs:
            values += [x + translation[0], y + translation[1]]
    params = " ".join(f"{v:.10g}" for v in values)
    return "\n".join(
        [
            "ctrl.png", "mapped.png", "0", "0", f"0 0 {width} {height}", f"0 0 {width} {height}",
            f"GridTransform_double_2_2 vp {len(values)} {params} fp 7 0 {grid_h - 1} {grid_w - 1} 0 0 {width} {height}",
        ]
    ) + "\n"


def write_field(path: Path, z: int, dx_nm: float, dy_nm: float, size: int = 4, pitch: float = 4000.0) -> None:
    """A constant residual over a ``size x size`` lattice from the origin (``pitch`` nm apart)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dx=np.full((size, size), dx_nm, dtype=np.float32),
        dy=np.full((size, size), dy_nm, dtype=np.float32),
        valid=np.ones((size, size), dtype=bool),
        n=np.full((size, size), 3, dtype=np.int32),
        z=np.int64(z),
        origin_xy_nm=np.array([0.0, 0.0]),
        pitch_nm=np.float64(pitch),
        kernel_radius_nm=np.float64(8000.0),
    )
    written = path.with_name(path.name + ".npz")  # np.savez appends .npz to a name without it
    if written.exists():
        written.replace(path)


def write_volume(
    root: Path,
    volume: str = "V",
    *,
    translations: dict[int, tuple[float, float]],
    reference: int = 10,
    zipped: bool = False,
    scale: float = 2.0,
) -> Path:
    """``<root>/<volume>/Vol.VikingXML`` with one grid stos per section of ``translations`` (mapped -> reference)."""
    folder = root / volume
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for z, shift in translations.items():
        name = f"{z}-{reference}.stos"
        rows.append(
            f'<stos controlSection="{reference}" mappedSection="{z}" path="{name}" pixelspacing="1" type="Grid"/>'
        )
        if zipped:
            with zipfile.ZipFile(folder / "Group.zip", "a") as archive:
                archive.writestr(name, stos_text(shift))
        else:
            (folder / name).write_text(stos_text(shift), encoding="utf-8")
    zip_attr = ' zip="Group.zip"' if zipped else ""
    xml = (
        f'<?xml version="1.0" ?>\n<Volume Name="{volume}">\n  <Scale UnitsOfMeasure="nm" UnitsPerPixel="{scale}"/>\n'
        f'  <StosGroup Name="{GROUP}"{zip_attr}>\n    ' + "\n    ".join(rows) + "\n  </StosGroup>\n</Volume>\n"
    )
    path = folder / "Vol.VikingXML"
    path.write_text(xml, encoding="utf-8")
    return path


def write_corrections(root: Path, volume: str, fields: dict[int, tuple[float, float]]) -> Path:
    folder = root / volume / GROUP
    folder.mkdir(parents=True, exist_ok=True)
    for z, (dx, dy) in fields.items():
        write_field(folder / f"{z}.vfield.npz", z, dx, dy)
    manifest = {
        "stos_group": GROUP,
        "provenance": {"built_utc": "2026-10-18T02:12:23Z", "pitch_nm": 4000, "kernel_radius_nm": 8000,
                       "method": "joint-second-difference-v1"},
        "volume_url": f"http://host/{volume}/Vol.VikingXML",
        "z": sorted(fields),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class GridStosTests(unittest.TestCase):
    def test_a_translation_grid_maps_forward_and_back(self) -> None:
        stos = GridStos.parse(stos_text((1000.0, 500.0)))
        self.assertEqual((stos.grid_w, stos.grid_h, stos.width, stos.height), (4, 3, 400.0, 300.0))
        pts = np.array([[0.0, 0.0], [123.4, 56.7], [399.0, 299.0], [200.0, 150.0]])
        out, ok = stos.forward(pts)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(out, pts + [1000.0, 500.0], atol=1e-6)
        back, ok = stos.inverse(out)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(back, pts, atol=1e-6)

    def test_points_off_the_grid_are_not_mapped(self) -> None:
        stos = GridStos.parse(stos_text((10.0, 0.0)))
        _out, ok = stos.forward(np.array([[-1.0, 5.0], [5.0, 301.0], [5.0, 5.0]]))
        self.assertEqual(ok.tolist(), [False, False, True])
        _back, ok = stos.inverse(np.array([[0.0, 5.0], [500.0, 5.0], [20.0, 5.0]]))
        self.assertEqual(ok.tolist(), [False, False, True])

    def test_a_warped_cell_is_piecewise_linear_over_its_two_triangles(self) -> None:
        text = stos_text((0.0, 0.0), width=100, height=100, grid=(2, 2))
        stos = GridStos.parse(text)
        stos.control[3] += [10.0, 10.0]  # move the top-right corner: only the upper triangle changes
        lower, _ = stos.forward(np.array([[20.0, 30.0]]))
        upper, _ = stos.forward(np.array([[80.0, 70.0]]))
        np.testing.assert_allclose(lower, [[20.0, 30.0]])
        self.assertGreater(float(upper[0, 0]), 80.0)
        back, ok = stos.inverse(upper)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(back, [[80.0, 70.0]], atol=1e-9)

    def test_non_grid_or_nan_transforms_are_refused(self) -> None:
        self.assertIsNone(GridStos.parse(stos_text((0.0, 0.0)).replace("GridTransform", "MeshTransform")))
        self.assertIsNone(GridStos.parse(stos_text((0.0, 0.0)).replace("fp 7", "vp 1 nan fp 7")))


class SectionFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "1.vfield.npz"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _field(self, dx: np.ndarray, valid: np.ndarray, origin=(0.0, 0.0)) -> SectionField:
        np.savez(
            self.path, dx=dx.astype(np.float32), dy=np.zeros_like(dx, dtype=np.float32), valid=valid,
            origin_xy_nm=np.array(origin, dtype=np.float64), pitch_nm=np.float64(1000.0),
            kernel_radius_nm=np.float64(3000.0), z=np.int64(1),
        )
        return SectionField.load(self.path)

    def test_bilinear_inside_renormalised_at_a_gap_inverse_distance_beyond_and_zero_far_away(self) -> None:
        dx = np.array([[0.0, 10.0], [20.0, 30.0]])
        field = self._field(dx, np.ones((2, 2), dtype=bool), origin=(1000.0, 1000.0))  # nodes at 1000 and 2000 nm
        out, trusted = field.sample(np.array([[1500.0, 1500.0], [1000.0, 1000.0], [1250.0, 1000.0]]))
        np.testing.assert_allclose(out[:, 0], [15.0, 0.0, 2.5])
        self.assertTrue(trusted.all())
        gap = self._field(dx, np.array([[True, True], [True, False]]), origin=(1000.0, 1000.0))
        out, _ = gap.sample(np.array([[1500.0, 1500.0]]))
        self.assertAlmostEqual(float(out[0, 0]), (0.0 + 10.0 + 20.0) / 3.0, places=5)
        out, trusted = field.sample(np.array([[4000.0, 1000.0]]))  # no corner present: inverse distance
        self.assertTrue(trusted[0])
        self.assertGreater(float(out[0, 0]), 5.0)
        out, trusted = field.sample(np.array([[50000.0, 50000.0]]))
        self.assertFalse(trusted[0])
        self.assertEqual(out[0].tolist(), [0.0, 0.0])

    def test_a_missing_file_is_none(self) -> None:
        self.assertIsNone(SectionField.load(Path(self._tmp.name) / "nope.npz"))


class ProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registration = self.root / "storage"
        self.corrections = self.root / "corrections"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_stos_groups_read_from_a_folder_or_a_zip_and_chains_are_flagged(self) -> None:
        for zipped in (False, True):
            xml = write_volume(self.registration / ("z" if zipped else "f"), translations={4: (100.0, 50.0)},
                               zipped=zipped)
            group = load_stos_group(xml, GROUP)
            self.assertEqual(sorted(group.entries), [4])
            self.assertIsInstance(group.transform(10), IdentityStos)  # the reference section
            self.assertIsNone(group.transform(7))  # neither mapped nor the reference
            out, ok = group.transform(4).forward(np.array([[1.0, 2.0]]))
            np.testing.assert_allclose(out, [[101.0, 52.0]])
            self.assertFalse(group.chained)
        self.assertEqual(volume_scale_nm(xml), 2.0)
        self.assertIsNone(load_stos_group(xml, "Other"))
        chained = self.registration / "c" / "V" / "Vol.VikingXML"
        chained.parent.mkdir(parents=True)
        chained.write_text(
            '<Volume><stos GroupName="g" controlSection="1" mappedSection="2" path="a.stos" pixelSpacing="16" type="grid"/>'
            '<stos GroupName="g" controlSection="2" mappedSection="3" path="b.stos" pixelSpacing="16" type="grid"/></Volume>',
            encoding="utf-8",
        )
        self.assertTrue(load_stos_group(chained, "g").chained)

    def test_projection_is_neighbour_stos_then_residual_then_inverse_target_stos(self) -> None:
        write_volume(self.registration, translations={4: (100.0, 50.0), 5: (40.0, 20.0)})
        write_corrections(self.corrections, "V", {4: (8.0, 0.0), 5: (0.0, 4.0)})  # nm: 4 px and 2 px at 2 nm/px
        projector, note = projector_for_volume("V", registration_root=self.registration,
                                               corrections_root=self.corrections)
        self.assertIsNotNone(projector, note)
        p = np.array([[10.0, 20.0], [30.0, 40.0]])
        stos_only, ok = projector.project(p, 4, 5, CONVENTION_NONE)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(stos_only, p + [60.0, 30.0], atol=1e-6)
        corrected, _ = projector.project(p, 4, 5)  # +n-t: v + f_4 - f_5
        np.testing.assert_allclose(corrected, p + [64.0, 28.0], atol=1e-5)
        flipped, _ = projector.project(p, 4, 5, "-n+t")
        np.testing.assert_allclose(flipped, p + [56.0, 32.0], atol=1e-5)

    def test_missing_data_gives_no_projector_and_says_why(self) -> None:
        projector, note = projector_for_volume("V", registration_root=None, corrections_root=None)
        self.assertIsNone(projector)
        self.assertIn("registration_root", note)
        projector, note = projector_for_volume("V", registration_root=self.registration, corrections_root=self.corrections)
        self.assertIsNone(projector)
        write_corrections(self.corrections, "V", {4: (0.0, 0.0)})
        projector, note = projector_for_volume("V", registration_root=self.registration, corrections_root=self.corrections)
        self.assertIsNone(projector)
        self.assertIn("not found", note)
        self.assertIsNotNone(find_correction_set(self.corrections, "V"))


class ProjectedDrawTests(unittest.TestCase):
    """Target 1 (z 5) is the box 100-140; neighbour 2 (z 4) lands on it through the stos (sections move by 60, 30)."""

    KEY = "V_5_D1_X64-192_Y64-192"

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_volume(self.root / "storage", translations={4: (100.0, 50.0), 5: (40.0, 20.0)})
        self.target_mask = np.zeros((128, 128), dtype=bool)
        self.target_mask[36:77, 36:77] = True  # 100-140 in mosaic px (edges included, as the outline is drawn), crop from 64

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _shapes(self, neighbour_box=(40, 70, 80, 110), *, fields=None, residual="auto"):
        records = [
            _record(1, 5, (100, 100, 140, 140), linked=[(2, 4)]),
            _record(2, 4, neighbour_box, has_mask=False, linked=[(1, 5)]),
        ]
        path = _write(self.root / f"shapes_{len(list(self.root.iterdir()))}.jsonl", records)
        shapes = load_volume_shapes(path)
        corrections = None
        if fields is not None:
            corrections = self.root / "corrections"
            write_corrections(corrections, "V", fields)
        projector, note = projector_for_volume("V", registration_root=self.root / "storage", corrections_root=corrections,
                                               stos_group=GROUP, vikingxml="Vol.VikingXML")
        self.assertIsNotNone(projector, note)
        outlines, stats = project_volume_shapes(shapes, projector, residual)
        shapes.projected = outlines
        return shapes, stats

    def test_the_projected_outline_lands_on_the_target(self) -> None:
        shapes, stats = self._shapes()
        self.assertEqual((stats["projected"], stats["failed"]), (1, 0))
        draw = draw_projected(shapes, self.KEY, 1, self.target_mask, np.random.default_rng(0))
        self.assertTrue(draw.projected and draw.passed)
        self.assertGreater(draw.overlap, 0.97)
        ys, xs = np.nonzero(draw.raster)
        self.assertLessEqual(abs(int(xs.min()) - 36), 1)
        self.assertLessEqual(abs(int(ys.max()) - 75), 1)

    def test_the_overlap_rule_fails_at_59_and_passes_at_61_percent(self) -> None:
        shapes, _ = self._shapes()
        raster = draw_projected(shapes, self.KEY, 1, self.target_mask).raster
        pixels = np.argwhere(raster)  # row-major: the mask covers the first rows of the outline
        for share, passed in ((0.59, False), (0.61, True)):
            mask = np.zeros_like(raster)
            keep = pixels[: int(round(share * len(pixels)))]
            mask[keep[:, 0], keep[:, 1]] = True
            draw = draw_projected(shapes, self.KEY, 1, mask, overlap_min=0.60)
            self.assertAlmostEqual(draw.overlap, share, delta=0.003)
            self.assertEqual(draw.passed, passed)

    def test_erosion_shrinks_the_outline_before_the_test(self) -> None:
        shapes, _ = self._shapes(neighbour_box=(30, 70, 70, 110))  # lands 10 px left of the target
        plain = draw_projected(shapes, self.KEY, 1, self.target_mask, np.random.default_rng(3))
        eroded = draw_projected(shapes, self.KEY, 1, self.target_mask, np.random.default_rng(3), erode_px=(8.0, 8.0))
        self.assertEqual(eroded.erode_px, 8.0)
        self.assertLess(int(eroded.raster.sum()), int(plain.raster.sum()))
        self.assertGreater(eroded.overlap, plain.overlap)
        ys, xs = np.nonzero(eroded.raster)
        self.assertGreaterEqual(int(xs.min()) - 26, 7)  # about 8 px in from the projected edge at 26
        drawn = draw_projected(shapes, self.KEY, 1, self.target_mask, np.random.default_rng(5), erode_px=(0.0, 10.0))
        self.assertTrue(0.0 <= drawn.erode_px <= 10.0)
        tiny = np.zeros((20, 20), dtype=bool)
        tiny[8:12, 8:12] = True
        self.assertTrue(erode_outline(tiny, 50.0).any())  # capped: never vanishes

    def test_the_draw_is_deterministic_for_a_seed(self) -> None:
        shapes, _ = self._shapes()
        first = draw_projected(shapes, self.KEY, 1, self.target_mask, np.random.default_rng([7, 2, 11, 0, 4]),
                               erode_px=(0.0, 10.0))
        again = draw_projected(shapes, self.KEY, 1, self.target_mask, np.random.default_rng([7, 2, 11, 0, 4]),
                               erode_px=(0.0, 10.0))
        self.assertEqual(first.erode_px, again.erode_px)
        np.testing.assert_array_equal(first.raster, again.raster)

    def test_no_projection_a_far_tile_or_an_unlinked_target_is_none(self) -> None:
        shapes, _ = self._shapes()
        self.assertIsNone(draw_projected(shapes, "V_5_D1_X2000-2128_Y2000-2128", 1, self.target_mask))
        self.assertIsNone(draw_projected(shapes, self.KEY, 2, self.target_mask))
        shapes.projected = None
        self.assertIsNone(draw_projected(shapes, self.KEY, 1, self.target_mask))

    def test_the_residual_is_kept_only_where_it_lowers_the_carry_error(self) -> None:
        # Without the residual the neighbour lands 4 px left and 2 px low; f_4 - f_5 = (+4, -2) px puts it on the target.
        _shapes, stats = self._shapes(neighbour_box=(36, 72, 76, 112), fields={4: (8.0, 0.0), 5: (0.0, 4.0)})
        self.assertEqual((stats["residual"], stats["convention"]), ("auto: on", "+n-t"))
        self.assertLess(stats["carry_median_px_residual"], stats["carry_median_px_stos"])
        _shapes, stats = self._shapes(fields={4: (8.0, 0.0), 5: (0.0, 4.0)})  # already on target: the residual hurts
        self.assertEqual((stats["residual"], stats["convention"]), ("auto: off (does not lower the carry error)", "none"))
        _shapes, stats = self._shapes(fields={4: (8.0, 0.0), 5: (0.0, 4.0)}, residual="on")
        self.assertEqual(stats["convention"], "+n-t")
        with self.assertRaises(ValueError):
            self._shapes(residual="maybe")


class TileContextProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_registration_data_falls_back_to_the_centroid_and_warns_once(self) -> None:
        records = [_record(1, 5, (10, 10, 30, 30), linked=[(2, 4)]), _record(2, 4, (10, 10, 20, 30), has_mask=False, linked=[(1, 5)])]
        shapes = load_volume_shapes(_write(self.root / SHAPES_FILENAME, records))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            record = attach_projection("NoData", shapes, registration_root=None, corrections_root=None)
            attach_projection("NoData", shapes, registration_root=None, corrections_root=None)
        self.assertEqual(record["placement"], "centroid")
        self.assertIn("registration_root", record["reason"])
        self.assertEqual(out.getvalue().count("WARNING"), 1)
        self.assertIsNone(shapes.projected)
        tiles = TileContext(shapes={"NoData": shapes})
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        draw = tiles.adjacent_draw(("NoData", "NoData_5_D1_X0-64_Y0-64"), 1, target=mask)
        self.assertIsInstance(draw, AdjacentDraw)
        self.assertFalse(draw.projected)
        self.assertTrue(draw.passed)
        self.assertIsNone(draw.overlap)

    def test_build_tile_context_records_the_projection_and_rejects_bad_settings(self) -> None:
        make_crops(self.root / "data", "V", n=2)
        examples = index_volumes(volumes=["V"], root=self.root / "data")
        with contextlib.redirect_stdout(io.StringIO()):
            context = build_tile_context(examples, load_circles=False)
        self.assertEqual(context.adjacent_placement, "projected")
        self.assertEqual(context.projection, {})  # make_crops writes no shapes.jsonl
        with self.assertRaises(ValueError):
            build_tile_context(examples, load_circles=False, adjacent_placement="volume")
        with self.assertRaises(ValueError):
            build_tile_context(examples, load_circles=False, adjacent_residual="sometimes")
        record = build_data_record(self.root / "data", ["V"], train_counts={}, val_counts={},
                                   adjacent_projection={"V": {"placement": "centroid", "reason": "x"}})
        self.assertEqual(record["adjacent_projection"]["V"]["placement"], "centroid")


class AdjacentPolyConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        cfg = PromptConfig.from_cfg({"prompt": {"adjacent_poly": {"overlap_min": 0.7, "erode_px": [2, 6]}}})
        self.assertEqual(cfg.adjacent_poly, AdjacentPoly(placement="projected", overlap_min=0.7, erode_px=(2.0, 6.0)))
        self.assertEqual(PromptConfig().adjacent_poly, AdjacentPoly())
        self.assertEqual(AdjacentPoly().erode_px, (0.0, 10.0))
        for bad in ({"placement": "volume"}, {"overlap_min": 0.0}, {"overlap_min": 1.5}, {"erode_px": [5, 2]},
                    {"erode_px": [-1, 2]}, {"overlap": 0.6}):
            with self.assertRaises(ValueError, msg=str(bad)):
                PromptConfig.from_cfg({"prompt": {"adjacent_poly": bad}})


class ProjectedPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_crops(self.root, "RC1", n=6)
        self.examples = index_volumes(volumes=["RC1"], root=self.root)
        self.base = build_tile_context(self.examples)
        # An outline that sticks out of the 8:24 target: a placed seed is used as it is, not fitted inside.
        self.outline = np.zeros((32, 32), dtype=bool)
        self.outline[4:20, 4:20] = True

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dataset(self, passed: bool, variants: int = 1) -> EMSegDataset:
        outline = self.outline

        class StubTiles(TileContext):
            def adjacent_draw(self, key, location_id, **_kwargs):
                return AdjacentDraw(raster=outline.copy(), projected=True, passed=passed, overlap=0.6 if passed else 0.3)

        values = {name: 0.0 for name in PromptConfig().__dataclass_fields__ if name.startswith("prob_")}
        values.update(prob_adjacent_poly_seed=0.5, prob_points_only=0.5, variants_per_sample=variants)
        return EMSegDataset(
            self.examples, split="val", image_size=32, seed=7,
            tiles=StubTiles(neighbors=self.base.neighbors, circles=self.base.circles),
            prompt_cfg=PromptConfig(**values),
        )

    def _adjacent_variant(self, ds: EMSegDataset) -> dict:
        for index in range(len(ds)):
            for variant in ds[index]["variants"]:
                if variant["requested_prompt_type"] == "adjacent_poly_seed":
                    return variant
        raise AssertionError("no variant asked for adjacent_poly_seed")

    def test_a_passing_outline_is_the_seed_where_it_lies(self) -> None:
        variant = self._adjacent_variant(self._dataset(True))
        self.assertEqual(variant["prompt_type"], "adjacent_poly_seed")
        self.assertEqual(variant["adjacent_rule"], "pass")
        self.assertAlmostEqual(variant["adjacent_overlap"], 0.6)
        np.testing.assert_array_equal(variant["prompt_input"].numpy().astype(bool), self.outline)

    def test_a_failing_outline_is_redrawn_as_another_family_and_logged(self) -> None:
        variant = self._adjacent_variant(self._dataset(False))
        # Drawn again from the other families, the adjacent share moved to the circle peers as when none exists.
        self.assertIn(variant["prompt_type"], {"points", "circle_seed"})
        self.assertEqual(variant["adjacent_rule"], "fail")
        ds = self._dataset(False, variants=2)
        for index in range(len(ds)):
            built = [v["prompt_type"] for v in ds[index]["variants"]]
            self.assertNotIn("adjacent_poly_seed", built)
            self.assertEqual(len(set(built)), len(built))

    def test_the_redraw_is_deterministic(self) -> None:
        ds = self._dataset(False, variants=2)
        plan = [PromptType.ADJACENT_POLY_SEED, PromptType.POINTS]
        self.assertEqual(
            ds._redraw_rejected_adjacent(ds.prompt_cfg, 3, 1, plan, full_tile=False),
            ds._redraw_rejected_adjacent(ds.prompt_cfg, 3, 1, plan, full_tile=False),
        )
        built, requested = ds._redraw_rejected_adjacent(ds.prompt_cfg, 3, 1, plan, full_tile=False)
        self.assertEqual(requested, plan)
        self.assertNotIn(PromptType.ADJACENT_POLY_SEED, built)

    def test_a_dropped_redraw_still_counts_its_failure_on_the_sample(self) -> None:
        from sam2_segmentation_trainer.data.dataset import collate_em

        outline = self.outline

        class StubTiles(TileContext):
            def adjacent_draw(self, key, location_id, **_kwargs):
                return AdjacentDraw(raster=outline.copy(), projected=True, passed=False, overlap=0.3)

        # adjacent and circle_seed share the mix and the shortfall adds points: a rejected outline's redraw finds
        # nothing new (its share moves to circle_seed, already planned), so its variant is dropped.
        values = {name: 0.0 for name in PromptConfig().__dataclass_fields__ if name.startswith("prob_")}
        values.update(prob_adjacent_poly_seed=0.5, prob_circle_seed=0.5, variants_per_sample=3)
        ds = EMSegDataset(
            self.examples, split="val", image_size=32, seed=7,
            tiles=StubTiles(neighbors=self.base.neighbors, circles=self.base.circles),
            prompt_cfg=PromptConfig(**values),
        )
        items = [ds[index] for index in range(len(ds))]
        for item in items:
            built = [v["prompt_type"] for v in item["variants"]]
            self.assertEqual(sorted(built), ["circle_seed", "points"])
            self.assertNotIn("adjacent_poly_seed", [v["requested_prompt_type"] for v in item["variants"]])
            # The variant is gone, but the sample keeps the failed rule and the dropped request.
            self.assertEqual(item["adjacent_rule"], "fail")
            self.assertEqual(item["dropped_prompt_types"], ["adjacent_poly_seed"])
        batch = collate_em(items[:2])
        self.assertEqual(batch["adjacent_rule"], ["fail", "fail"])  # per sample
        self.assertEqual(batch["dropped_prompt_type"], ["adjacent_poly_seed", "adjacent_poly_seed"])
        passing = self._dataset(True)
        self.assertTrue(all(passing[i]["dropped_prompt_types"] == [] for i in range(len(passing))))

    def test_a_placed_seeds_click_lies_on_the_target(self) -> None:
        target = np.zeros((32, 32), dtype=bool)
        target[8:24, 8:24] = True
        # Mostly off the target: the seed's deepest pixel (about x 5.5) is outside it.
        outline = np.zeros((32, 32), dtype=bool)
        outline[0:20, 0:12] = True
        kwargs = dict(target_mask=target, circle_mask=None, regions=[], cfg=PromptConfig())
        for seed in range(5):
            placed = build_prompt(PromptType.ADJACENT_POLY_SEED, rng=np.random.default_rng(seed), adjacent_shape=outline,
                                  adjacent_placed=True, **kwargs)
            (x, y), = placed.point_coords
            self.assertTrue(target[int(y), int(x)], (x, y))
            self.assertTrue(outline[int(y), int(x)], (x, y))
            np.testing.assert_array_equal(placed.input_mask, outline)  # the seed itself is not clipped
        # A placed seed that misses the target entirely keeps a click on the seed.
        away = np.zeros((32, 32), dtype=bool)
        away[0:6, 0:6] = True
        missed = build_prompt(PromptType.ADJACENT_POLY_SEED, rng=np.random.default_rng(0), adjacent_shape=away,
                              adjacent_placed=True, **kwargs)
        (x, y), = missed.point_coords
        self.assertTrue(away[int(y), int(x)])

    def test_build_prompt_keeps_a_placed_seed_and_fits_an_unplaced_one(self) -> None:
        target = np.zeros((32, 32), dtype=bool)
        target[8:24, 8:24] = True
        kwargs = dict(target_mask=target, circle_mask=None, regions=[], cfg=PromptConfig())
        placed = build_prompt(PromptType.ADJACENT_POLY_SEED, rng=np.random.default_rng(0), adjacent_shape=self.outline,
                              adjacent_placed=True, **kwargs)
        np.testing.assert_array_equal(placed.input_mask, self.outline)
        self.assertFalse((placed.foreground_weight[~target] > 0).any())  # trust only where it overlaps the target
        fitted = build_prompt(PromptType.ADJACENT_POLY_SEED, rng=np.random.default_rng(0), adjacent_shape=self.outline,
                              **kwargs)
        self.assertFalse(np.array_equal(fitted.input_mask, self.outline))  # the centroid fallback scales it to fit


if __name__ == "__main__":
    unittest.main()
