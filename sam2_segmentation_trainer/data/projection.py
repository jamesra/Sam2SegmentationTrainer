"""Section-to-section projection of adjacent-section outlines, for the ``adjacent_poly_seed`` prompt family.

The exporter's ``shapes.jsonl`` stores every outline in its own section's MOSAIC pixels (unregistered; a linked
neighbour lies a median 9,000-23,000 px from its target), so an outline has to be carried through the volume's
registration before it can be drawn in the target's crop, the way an annotator's carried-forward polygon lands in
Viking:

1. neighbour mosaic -> volume, through the neighbour section's slice-to-volume transform (the stos group the Viking
   volume and the correction service use, e.g. ``SliceToVolume1``; ``Viking.VolumeModel.VolumeTransformProvider``);
2. the published residual correction (``SectionCorrection``; a per-section lattice of offsets in nm at
   ``pitch_nm``), applied in volume space as ``v + f_neighbour(v) - f_target(v)`` (:data:`CONVENTION`, chosen
   empirically on linked pairs: docs/mask_refinement_plan.md, section 3.1);
3. volume -> target mosaic, through the INVERSE of the target section's transform.

Where the data comes from (all read-only):

* ``data.registration_root``: the folder holding one subfolder per volume with its VikingXML and stos files, e.g.
  ``/storage4`` (the volume server's ``Data/Volumes`` share): ``<root>/<VOL>/<file>.VikingXML`` lists the stos group
  (``<StosGroup Name=... zip=...><stos .../></StosGroup>`` or ``<stos GroupName=... />``) and each ``.stos`` path,
  read from the group's zip when there is one, else from the folder.
* ``data.corrections_root``: the correction service's published sets, ``<root>/<VOL>/<stos group>/manifest.json``
  plus ``<z>.vfield.npz`` per section (``Viking.SectionCorrection.PublishedCorrectionSet``). The manifest names the
  stos group and the VikingXML (``volume_url``); its ``provenance`` is recorded in the run's ``training_data.json``.

Only grid stos transforms (``GridTransform_double_2_2``) are read, and only groups in which every section maps
straight to one reference section (RC2, RPC1, RPC2 ``SliceToVolume1``); a chained group (RC1 ``grid16``, each
section registered to the one before) is reported and that volume keeps the centroid placement.

Everything is numpy; :func:`project_volume_shapes` runs once in the main process and stores, per (target,
neighbour) link, the neighbour's rings already in the target's mosaic pixels, so the DataLoader workers inherit a
few flat arrays and never touch a transform.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree

CONVENTION_NONE = "none"
"""Stos only: no residual correction."""

CONVENTIONS = {
    CONVENTION_NONE: (0.0, 0.0),
    "+n": (1.0, 0.0),
    "-n": (-1.0, 0.0),
    "+t": (0.0, 1.0),
    "-t": (0.0, -1.0),
    "+n-t": (1.0, -1.0),
    "-n+t": (-1.0, 1.0),
}
"""Residual conventions: ``(a, b)`` in ``v' = v + a * f_neighbour(v) + b * f_target(v)`` (volume nm)."""

CONVENTION = "+n-t"
"""The convention the trainer uses (the residual is the offset that moves a section's volume position onto the
corrected volume: forward through the neighbour's, back through the target's)."""


# ----------------------------------------------------------------------------------------------------------------
# stos grid transforms


def _parse_parameters(line: str) -> tuple[str, np.ndarray, np.ndarray]:
    """``(name, variable, fixed)`` from an ITK transform line ``Name vp N v... fp M f...``."""
    parts = line.split()
    name = parts[0]
    i = 1
    variable = fixed = np.zeros(0)
    while i < len(parts):
        tag = parts[i]
        count = int(float(parts[i + 1]))
        values = np.array([float(v) for v in parts[i + 2 : i + 2 + count]], dtype=np.float64)
        if tag == "vp":
            variable = values
        elif tag == "fp":
            fixed = values
        i += 2 + count
    return name, variable, fixed


@dataclass(slots=True)
class GridStos:
    """A grid stos transform (``GridTransform_double_2_2``): a regular ``grid_w x grid_h`` lattice over the mapped
    section's image, ``(0..width) x (0..height)``, each node paired with its control (volume) position; two triangles
    per cell, as ``Geometry.Transforms.TransformFactory.ParseGridTransform`` builds them. Forward maps mapped
    (section mosaic) points to control (volume) points; inverse maps back. Points no triangle covers are not mapped."""

    control: np.ndarray  # (grid_h * grid_w, 2) float64, row-major (x fastest)
    grid_w: int
    grid_h: int
    width: float
    height: float
    _tree: cKDTree | None = field(default=None, repr=False)
    _triangles: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def parse(cls, text: str, pixel_spacing: float = 1.0) -> "GridStos | None":
        """From a ``.stos`` file's text; None when the transform is not a grid or holds NaN."""
        lines = text.splitlines()
        if len(lines) < 7 or "nan" in lines[6].lower():
            return None
        name, variable, fixed = _parse_parameters(lines[6])
        if name.lower() != "gridtransform_double_2_2" or fixed.size < 7:
            return None
        grid_w = int(round(fixed[2] + 1.0))
        grid_h = int(round(fixed[1] + 1.0))
        if variable.size != 2 * grid_w * grid_h or grid_w < 2 or grid_h < 2:
            return None
        return cls(
            control=variable.reshape(-1, 2) * float(pixel_spacing),
            grid_w=grid_w,
            grid_h=grid_h,
            width=float(round(fixed[5] * pixel_spacing)),
            height=float(round(fixed[6] * pixel_spacing)),
        )

    def mapped_nodes(self) -> np.ndarray:
        """The lattice nodes in mapped (section) pixels, row-major like :attr:`control`."""
        xs = np.arange(self.grid_w) / (self.grid_w - 1) * self.width
        ys = np.arange(self.grid_h) / (self.grid_h - 1) * self.height
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    def forward(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mapped -> control. ``(out (n, 2), ok (n,))``; points outside the grid are not mapped (ok False)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        fx = pts[:, 0] / self.width * (self.grid_w - 1)
        fy = pts[:, 1] / self.height * (self.grid_h - 1)
        ok = (fx >= 0) & (fx <= self.grid_w - 1) & (fy >= 0) & (fy <= self.grid_h - 1)
        ix = np.clip(np.floor(fx).astype(np.int64), 0, self.grid_w - 2)
        iy = np.clip(np.floor(fy).astype(np.int64), 0, self.grid_h - 2)
        tx = fx - ix
        ty = fy - iy
        c = self.control
        bl = c[ix + iy * self.grid_w]
        br = c[ix + 1 + iy * self.grid_w]
        tl = c[ix + (iy + 1) * self.grid_w]
        tr = c[ix + 1 + (iy + 1) * self.grid_w]
        lower = (tx + ty) <= 1.0
        out = np.where(
            lower[:, None],
            bl + tx[:, None] * (br - bl) + ty[:, None] * (tl - bl),
            tr + (1.0 - tx)[:, None] * (tl - tr) + (1.0 - ty)[:, None] * (br - tr),
        )
        return out, ok

    def _triangle_index(self) -> tuple[cKDTree, np.ndarray]:
        if self._tree is None or self._triangles is None:
            w = self.grid_w
            x, y = np.meshgrid(np.arange(w - 1), np.arange(self.grid_h - 1))
            bl = (x + y * w).ravel()
            br = bl + 1
            tl = bl + w
            tr = tl + 1
            tris = np.concatenate([np.stack([bl, br, tl], 1), np.stack([br, tr, tl], 1)])
            self._triangles = tris
            self._tree = cKDTree(self.control[tris].mean(axis=1))
        return self._tree, self._triangles

    def inverse(self, points: np.ndarray, candidates: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """Control -> mapped, through the same triangles. ``(out (n, 2), ok (n,))``; a point no triangle among the
        ``candidates`` nearest (by centroid) contains is not mapped."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.zeros_like(pts)
        ok = np.zeros(len(pts), dtype=bool)
        if not len(pts):
            return out, ok
        tree, tris = self._triangle_index()
        k = min(candidates, len(tris))
        _dist, near = tree.query(pts, k=k)
        near = np.asarray(near).reshape(len(pts), k)
        mapped = self.mapped_nodes()
        for j in range(k):
            todo = ~ok
            if not todo.any():
                break
            t = tris[near[todo, j]]
            a, b, c = self.control[t[:, 0]], self.control[t[:, 1]], self.control[t[:, 2]]
            p = pts[todo]
            v0, v1, v2 = b - a, c - a, p - a
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            safe = np.abs(den) > 1e-12
            den = np.where(safe, den, 1.0)
            u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            eps = 1e-9
            inside = safe & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)
            ma, mb, mc = mapped[t[:, 0]], mapped[t[:, 1]], mapped[t[:, 2]]
            res = ma + u[:, None] * (mb - ma) + v[:, None] * (mc - ma)
            index = np.flatnonzero(todo)[inside]
            out[index] = res[inside]
            ok[index] = True
        return out, ok


class IdentityStos:
    """The reference section of a group (it has no stos: its mosaic is the volume)."""

    def forward(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return pts.copy(), np.ones(len(pts), dtype=bool)

    inverse = forward


@dataclass(slots=True)
class StosEntry:
    control: int
    mapped: int
    path: str
    pixel_spacing: float
    kind: str


@dataclass
class StosGroup:
    """One volume's stos group from its VikingXML; transforms are read on first use and kept."""

    name: str
    volume_dir: Path
    entries: dict[int, StosEntry]
    zip_path: Path | None = None
    source: Path | None = None
    """The VikingXML the group was read from."""
    _cache: dict[int, object] = field(default_factory=dict, repr=False)
    _zip: zipfile.ZipFile | None = field(default=None, repr=False)

    @property
    def reference_sections(self) -> set[int]:
        return {e.control for e in self.entries.values()}

    @property
    def chained(self) -> bool:
        """True when some control section is itself mapped (a chain, not one reference); unsupported."""
        return any(control in self.entries for control in self.reference_sections)

    def _read(self, entry: StosEntry) -> str | None:
        rel = PureWindowsPath(entry.path).as_posix() if "\\" in entry.path else entry.path
        if self.zip_path is not None:
            try:
                if self._zip is None:
                    self._zip = zipfile.ZipFile(self.zip_path)
                names = set(self._zip.namelist())
                for name in (rel, PurePosixPath(rel).name):
                    if name in names:
                        return self._zip.read(name).decode("utf-8", errors="replace")
            except (OSError, zipfile.BadZipFile):
                pass
        candidates = [self.volume_dir / rel]
        if self.zip_path is not None:
            candidates.append(self.zip_path.with_suffix("") / PurePosixPath(rel).name)
        for path in candidates:
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return None

    def transform(self, z: int):
        """The section's transform (mosaic -> volume), :class:`IdentityStos` for the reference section, or None."""
        z = int(z)
        if z in self._cache:
            return self._cache[z]
        entry = self.entries.get(z)
        result: object | None
        if entry is None:
            result = IdentityStos() if z in self.reference_sections else None
        else:
            text = self._read(entry)
            result = None if text is None else GridStos.parse(text, entry.pixel_spacing)
        self._cache[z] = result
        return result

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None


def _attr(elem: ET.Element, name: str) -> str | None:
    lowered = name.lower()
    for key, value in elem.attrib.items():
        if key.lower() == lowered:
            return value
    return None


def load_stos_group(vikingxml: Path, group: str) -> StosGroup | None:
    """The stos group ``group`` of a VikingXML file, or None when the file or the group is missing."""
    try:
        root = ET.parse(vikingxml).getroot()
    except (OSError, ET.ParseError):
        return None
    volume_dir = Path(vikingxml).parent
    entries: dict[int, StosEntry] = {}
    zip_path: Path | None = None

    def add(elem: ET.Element) -> None:
        try:
            entry = StosEntry(
                control=int(_attr(elem, "controlSection")),
                mapped=int(_attr(elem, "mappedSection")),
                path=str(_attr(elem, "path")),
                pixel_spacing=float(_attr(elem, "pixelSpacing") or 1.0),
                kind=str(_attr(elem, "type") or ""),
            )
        except (TypeError, ValueError):
            return
        entries.setdefault(entry.mapped, entry)

    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "StosGroup" and _attr(elem, "Name") == group:
            zip_attr = _attr(elem, "zip")
            if zip_attr:
                zip_path = volume_dir / PureWindowsPath(zip_attr).as_posix()
            for child in elem:
                if child.tag.rsplit("}", 1)[-1] == "stos":
                    add(child)
        elif tag == "stos" and _attr(elem, "GroupName") == group:
            add(elem)
    if not entries:
        return None
    return StosGroup(name=group, volume_dir=volume_dir, entries=entries, zip_path=zip_path, source=Path(vikingxml))


# ----------------------------------------------------------------------------------------------------------------
# residual correction fields


@dataclass(slots=True)
class SectionField:
    """One section's residual lattice (``Viking.SectionCorrection.SectionVectorField``): node ``(gx, gy)`` sits at
    ``(gx * pitch, gy * pitch)`` volume nm; ``dx``/``dy`` (nm, float32) are valid where ``valid``."""

    gx0: int
    gy0: int
    pitch: float
    kernel: float
    dx: np.ndarray
    dy: np.ndarray
    valid: np.ndarray

    @classmethod
    def load(cls, path: Path, pitch_default: float = 4000.0, kernel_default: float = 8000.0) -> "SectionField | None":
        try:
            with np.load(path) as data:
                pitch = float(data["pitch_nm"]) if "pitch_nm" in data else pitch_default
                kernel = float(data["kernel_radius_nm"]) if "kernel_radius_nm" in data else kernel_default
                origin = np.asarray(data["origin_xy_nm"], dtype=np.float64) if "origin_xy_nm" in data else np.zeros(2)
                dx = np.asarray(data["dx"], dtype=np.float32)
                dy = np.asarray(data["dy"], dtype=np.float32)
                valid = np.asarray(data["valid"], dtype=bool)
        except (OSError, KeyError, ValueError):
            return None
        if dx.ndim != 2:
            dx, dy, valid = dx.reshape(1, -1), dy.reshape(1, -1), valid.reshape(1, -1)
        return cls(
            gx0=int(round(origin[0] / pitch)) if pitch > 0 else 0,
            gy0=int(round(origin[1] / pitch)) if pitch > 0 and origin.size > 1 else 0,
            pitch=pitch if pitch > 0 else 2000.0,
            kernel=kernel if kernel > 0 else 8000.0,
            dx=dx,
            dy=dy,
            valid=valid,
        )

    def _lookup(self, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = gx - self.gx0
        iy = gy - self.gy0
        h, w = self.valid.shape
        inside = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
        ixc = np.clip(ix, 0, w - 1)
        iyc = np.clip(iy, 0, h - 1)
        ok = inside & self.valid[iyc, ixc]
        return ok, np.where(ok, self.dx[iyc, ixc], 0.0), np.where(ok, self.dy[iyc, ixc], 0.0)

    def sample(self, xy_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Offsets (nm, ``(n, 2)``) at volume-nm points and whether each is trusted, as ``SectionVectorField.Sample``:
        bilinear over the lattice corners present (renormalised), else inverse-distance over nodes within the kernel
        radius, else zero (untrusted)."""
        pts = np.asarray(xy_nm, dtype=np.float64).reshape(-1, 2)
        fx = pts[:, 0] / self.pitch
        fy = pts[:, 1] / self.pitch
        gx = np.floor(fx).astype(np.int64)
        gy = np.floor(fy).astype(np.int64)
        tx = fx - gx
        ty = fy - gy
        acc = np.zeros_like(pts)
        ws = np.zeros(len(pts))
        for ox, oy, q in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
            ok, dx, dy = self._lookup(gx + ox, gy + oy)
            q = np.where(ok, q, 0.0)
            acc[:, 0] += dx * q
            acc[:, 1] += dy * q
            ws += q
        out = np.zeros_like(pts)
        trusted = ws > 1e-6
        out[trusted] = acc[trusted] / ws[trusted, None]
        rest = np.flatnonzero(~trusted)
        if rest.size:
            reach = int(math.ceil(self.kernel / self.pitch)) + 1
            r2 = self.kernel * self.kernel
            wsum = np.zeros(rest.size)
            wacc = np.zeros((rest.size, 2))
            p = pts[rest]
            for ox in range(-reach, reach + 1):
                for oy in range(-reach, reach + 1):
                    nx, ny = gx[rest] + ox, gy[rest] + oy
                    ok, dx, dy = self._lookup(nx, ny)
                    d2 = (p[:, 0] - nx * self.pitch) ** 2 + (p[:, 1] - ny * self.pitch) ** 2
                    use = ok & (d2 <= r2)
                    w = np.where(use, 1.0 / (d2 + 1.0), 0.0)
                    wacc[:, 0] += dx * w
                    wacc[:, 1] += dy * w
                    wsum += w
            good = wsum > 1e-12
            out[rest[good]] = wacc[good] / wsum[good, None]
            trusted[rest[good]] = True
        return out, trusted


@dataclass
class CorrectionSet:
    """One published correction set (``<corrections_root>/<VOL>/<stos group>/``); sections load on first use."""

    directory: Path
    manifest: dict
    _cache: dict[int, SectionField | None] = field(default_factory=dict, repr=False)

    @property
    def stos_group(self) -> str:
        return str(self.manifest.get("stos_group") or self.directory.name)

    @property
    def provenance(self) -> dict:
        return dict(self.manifest.get("provenance") or {})

    def section(self, z: int) -> SectionField | None:
        z = int(z)
        if z not in self._cache:
            prov = self.provenance
            self._cache[z] = SectionField.load(
                self.directory / f"{z}.vfield.npz",
                float(prov.get("pitch_nm") or 4000.0),
                float(prov.get("kernel_radius_nm") or 8000.0),
            )
        return self._cache[z]

    def offset_px(self, z: int, volume_px: np.ndarray, nm_per_px: float) -> np.ndarray:
        """The section's residual at volume-pixel points, in volume pixels (zero where it has no field)."""
        pts = np.asarray(volume_px, dtype=np.float64).reshape(-1, 2)
        sec = self.section(z)
        if sec is None:
            return np.zeros_like(pts)
        offset, _trusted = sec.sample(pts * nm_per_px)
        return offset / nm_per_px


def find_correction_set(corrections_root: Path, volume: str, group: str | None = None) -> CorrectionSet | None:
    """The volume's published set: ``<root>/<volume>/<group>/manifest.json`` (with no ``group``, the only one)."""
    base = Path(corrections_root) / volume
    if group is not None:
        folders = [base / group]
    else:
        try:
            folders = sorted(p for p in base.iterdir() if (p / "manifest.json").is_file())
        except OSError:
            return None
        if len(folders) != 1:
            return None
    try:
        manifest = json.loads((folders[0] / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return CorrectionSet(directory=folders[0], manifest=manifest)


_SCALE_RE = re.compile(r"<Scale\b[^>]*UnitsPerPixel=\"(?P<v>[0-9.eE+-]+)\"", re.IGNORECASE)


def volume_scale_nm(vikingxml: Path) -> float | None:
    """The volume's pixel size (nm), from the ``<Scale>`` directly under ``<Volume>``."""
    try:
        root = ET.parse(vikingxml).getroot()
    except (OSError, ET.ParseError):
        return None
    for child in root:
        if child.tag.rsplit("}", 1)[-1] == "Scale":
            try:
                return float(_attr(child, "UnitsPerPixel"))
            except (TypeError, ValueError):
                return None
    return None


# ----------------------------------------------------------------------------------------------------------------
# the projector


@dataclass
class SectionProjector:
    """Maps points from one section's mosaic pixels to another's: stos forward, residual, stos inverse."""

    stos: StosGroup
    corrections: CorrectionSet | None
    nm_per_px: float
    convention: str = CONVENTION

    def to_volume(self, z: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = self.stos.transform(z)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if transform is None:
            return np.zeros_like(pts), np.zeros(len(pts), dtype=bool)
        return transform.forward(pts)

    def from_volume(self, z: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = self.stos.transform(z)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if transform is None:
            return np.zeros_like(pts), np.zeros(len(pts), dtype=bool)
        return transform.inverse(pts)

    def correct(self, volume_px: np.ndarray, z_from: int, z_to: int, convention: str | None = None) -> np.ndarray:
        a, b = CONVENTIONS[convention or self.convention]
        out = np.asarray(volume_px, dtype=np.float64).reshape(-1, 2).copy()
        if self.corrections is None or (a == 0.0 and b == 0.0):
            return out
        base = out.copy()
        if a:
            out += a * self.corrections.offset_px(z_from, base, self.nm_per_px)
        if b:
            out += b * self.corrections.offset_px(z_to, base, self.nm_per_px)
        return out

    def project(
        self, points: np.ndarray, z_from: int, z_to: int, convention: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``points`` (mosaic px of ``z_from``) in mosaic px of ``z_to``; ``ok`` is False where a step failed."""
        volume, ok1 = self.to_volume(z_from, points)
        volume = self.correct(volume, z_from, z_to, convention)
        out, ok2 = self.from_volume(z_to, volume)
        return out, ok1 & ok2


@dataclass(slots=True)
class ProjectedOutlines:
    """Neighbour outlines already carried into their targets' mosaic frames, packed like
    :class:`~sam2_segmentation_trainer.data.shapes.VolumeShapes`: entry ``e`` has rings
    ``ring_start[e]:ring_start[e + 1]``, ring ``i`` is ``anchor[e] + points[ring_offsets[i]:ring_offsets[i + 1]]``
    (float32 offsets from a float64 anchor, so a 100,000 px coordinate keeps sub-pixel precision).
    ``by_target[t]`` lists the entry of each of ``VolumeShapes.links[t]`` in the same order, -1 where the projection
    failed (a point outside a stos grid)."""

    by_target: dict[int, tuple[int, ...]]
    anchor: np.ndarray
    ring_start: np.ndarray
    ring_offsets: np.ndarray
    ring_exterior: np.ndarray
    points: np.ndarray

    def __len__(self) -> int:
        return int(self.anchor.shape[0])

    def entries(self, location_id: int) -> tuple[int, ...]:
        return self.by_target.get(int(location_id), ())

    def rings(self, entry: int) -> list[tuple[np.ndarray, bool]]:
        base = self.anchor[entry]
        out: list[tuple[np.ndarray, bool]] = []
        for ring in range(int(self.ring_start[entry]), int(self.ring_start[entry + 1])):
            pts = self.points[self.ring_offsets[ring] : self.ring_offsets[ring + 1]].astype(np.float64) + base
            out.append((pts, bool(self.ring_exterior[ring])))
        return out


RESIDUAL_AUTO = "auto"
RESIDUAL_ON = "on"
RESIDUAL_OFF = "off"
RESIDUAL_MODES = (RESIDUAL_AUTO, RESIDUAL_ON, RESIDUAL_OFF)
"""``data.adjacent_residual``: ``auto`` applies the residual correction to a volume only when it lowers the median
distance between each linked neighbour's carried centroid and its target's centroid (measured over the volume's own
links at load time); ``on`` always (when published), ``off`` never."""


def _carry_errors(shapes, projector: SectionProjector, convention: str) -> np.ndarray:
    """Distance (target mosaic px) from each linked neighbour's carried centroid to its target's centroid."""
    out: list[float] = []
    for target, rows in shapes.links.items():
        zt = shapes.target_z.get(target)
        centre = shapes.target_centroids.get(target)
        if zt is None or centre is None:
            continue
        for row in rows:
            q, ok = projector.project(shapes.centroid[row][None, :], int(shapes.z[row]), int(zt), convention)
            if ok.all():
                out.append(float(np.hypot(*(q[0] - np.asarray(centre)))))
    return np.asarray(out, dtype=np.float64)


def project_volume_shapes(
    shapes, projector: SectionProjector, residual: str = RESIDUAL_AUTO
) -> tuple[ProjectedOutlines, dict]:
    """Carry every linked neighbour outline of ``shapes`` into its target's mosaic frame.

    Returns the packed outlines and a summary for the run record: the convention used, the residual decision and
    the median centroid carry error with and without it, and how many links projected.
    """
    if residual not in RESIDUAL_MODES:
        raise ValueError(f"data.adjacent_residual must be one of {RESIDUAL_MODES}, got {residual!r}")
    stats: dict = {"links": int(sum(len(rows) for rows in shapes.links.values()))}
    convention = projector.convention
    if projector.corrections is None or convention == CONVENTION_NONE:
        convention = CONVENTION_NONE
        stats["residual"] = "none published"
    elif residual == RESIDUAL_OFF:
        convention = CONVENTION_NONE
        stats["residual"] = "off"
    else:
        raw = _carry_errors(shapes, projector, CONVENTION_NONE)
        corrected = _carry_errors(shapes, projector, convention)
        stats["carry_median_px_stos"] = float(np.median(raw)) if raw.size else None
        stats["carry_median_px_residual"] = float(np.median(corrected)) if corrected.size else None
        better = raw.size > 0 and corrected.size > 0 and float(np.median(corrected)) < float(np.median(raw))
        if residual == RESIDUAL_AUTO and not better:
            convention = CONVENTION_NONE
            stats["residual"] = "auto: off (does not lower the carry error)"
        else:
            stats["residual"] = "on" if residual == RESIDUAL_ON else "auto: on"
    stats["convention"] = convention

    by_target: dict[int, tuple[int, ...]] = {}
    anchors: list[np.ndarray] = []
    ring_counts: list[int] = []
    lengths: list[int] = []
    exterior: list[bool] = []
    chunks: list[np.ndarray] = []
    failed = 0
    for target in sorted(shapes.links):
        zt = shapes.target_z.get(target)
        entries: list[int] = []
        for row in shapes.links[target]:
            rings = shapes.rings(row)
            if zt is None or not rings:
                entries.append(-1)
                failed += 1
                continue
            pts = np.concatenate([r for r, _e in rings])
            moved, ok = projector.project(pts, int(shapes.z[row]), int(zt), convention)
            if not ok.all():
                entries.append(-1)
                failed += 1
                continue
            anchor = moved.mean(axis=0)
            entries.append(len(anchors))
            anchors.append(anchor)
            ring_counts.append(len(rings))
            start = 0
            for ring, is_exterior in rings:
                piece = (moved[start : start + len(ring)] - anchor).astype(np.float32)
                start += len(ring)
                chunks.append(piece)
                lengths.append(len(piece))
                exterior.append(is_exterior)
        by_target[target] = tuple(entries)
    stats["projected"] = len(anchors)
    stats["failed"] = failed
    outlines = ProjectedOutlines(
        by_target=by_target,
        anchor=np.array(anchors, dtype=np.float64).reshape(-1, 2),
        ring_start=np.concatenate([[0], np.cumsum(ring_counts, dtype=np.int64)]).astype(np.int64),
        ring_offsets=np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64),
        ring_exterior=np.array(exterior, dtype=bool),
        points=np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32),
    )
    return outlines, stats


def projector_for_volume(
    volume: str,
    *,
    registration_root: Path | None,
    corrections_root: Path | None,
    stos_group: str | None = None,
    vikingxml: str | None = None,
    nm_per_px: float | None = None,
    convention: str = CONVENTION,
) -> tuple[SectionProjector | None, str]:
    """A projector for one volume and a one-line note of where it came from (or why there is none).

    The correction set (``<corrections_root>/<volume>/<group>/``) names the stos group and the VikingXML
    (``volume_url``'s file name) unless ``stos_group`` / ``vikingxml`` are given; the VikingXML is
    ``<registration_root>/<volume>/<file>``. Without a correction set the stos is still used when ``stos_group`` and
    ``vikingxml`` are given (convention ``none``).
    """
    if registration_root is None:
        return None, "no data.registration_root"
    corrections = (
        find_correction_set(Path(corrections_root), volume, stos_group) if corrections_root is not None else None
    )
    group = stos_group or (corrections.stos_group if corrections is not None else None)
    xml_name = vikingxml
    if xml_name is None and corrections is not None:
        url = str(corrections.manifest.get("volume_url") or "")
        xml_name = url.rstrip("/").rsplit("/", 1)[-1] or None
    if group is None or xml_name is None:
        return None, f"no correction set under {corrections_root} and no stos group/VikingXML configured"
    xml_path = Path(registration_root) / volume / xml_name
    if not xml_path.is_file():
        # The VikingXML name on the volume server is not always the one in the URL (case, or a renamed file).
        matches = [p for p in (Path(registration_root) / volume).glob("*") if p.name.lower() == xml_name.lower()]
        if not matches:
            return None, f"{xml_path} not found"
        xml_path = matches[0]
    stos = load_stos_group(xml_path, group)
    if stos is None:
        return None, f"stos group {group} not in {xml_path}"
    if stos.chained:
        return None, f"stos group {group} in {xml_path} is chained (not one reference section); unsupported"
    scale = nm_per_px or volume_scale_nm(xml_path)
    if scale is None or scale <= 0:
        return None, f"no volume <Scale> in {xml_path}"
    if corrections is None:
        convention = CONVENTION_NONE
    note = (
        f"{xml_path} group {group} ({len(stos.entries)} stos, reference {sorted(stos.reference_sections)}), "
        f"{scale} nm/px, residual "
        + (f"{corrections.directory} built {corrections.provenance.get('built_utc')}" if corrections else "none")
        + f", convention {convention}"
    )
    return SectionProjector(stos=stos, corrections=corrections, nm_per_px=float(scale), convention=convention), note
