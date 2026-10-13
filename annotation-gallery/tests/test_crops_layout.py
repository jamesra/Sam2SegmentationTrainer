"""crops_dir_for accepts the flat training layout and the legacy nested one."""

from __future__ import annotations

import sys
from pathlib import Path

GALLERY_ROOT = Path(__file__).resolve().parents[1]
if str(GALLERY_ROOT) not in sys.path:
    sys.path.insert(0, str(GALLERY_ROOT))

from crops_path import crops_dir_for  # noqa: E402
from server import crops_dir  # noqa: E402
from thumbs import registry_signature  # noqa: E402


def test_flat_volume_uses_volume_root(tmp_path: Path) -> None:
    volume = tmp_path / "RC1"
    volume.mkdir()
    (volume / "manifest.jsonl").write_text("", encoding="utf-8")
    assert crops_dir_for(volume) == volume
    assert crops_dir(tmp_path, "RC1") == volume


def test_nested_volume_uses_annotation_crops(tmp_path: Path) -> None:
    volume = tmp_path / "RC1"
    (volume / "AnnotationCrops").mkdir(parents=True)
    assert crops_dir_for(volume) == volume / "AnnotationCrops"
    assert crops_dir(tmp_path, "RC1") == volume / "AnnotationCrops"


def test_flat_wins_when_both_exist(tmp_path: Path) -> None:
    volume = tmp_path / "RC1"
    (volume / "AnnotationCrops").mkdir(parents=True)
    (volume / "annotation_crops.sqlite").write_bytes(b"")
    assert crops_dir_for(volume) == volume


def test_registry_signature_sees_flat_and_nested(tmp_path: Path) -> None:
    flat = tmp_path / "RC1"
    (flat / "images").mkdir(parents=True)
    (flat / "manifest.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "RC2" / "AnnotationCrops" / "images").mkdir(parents=True)
    _link, rows = registry_signature(tmp_path)
    assert [name for name, _times in rows] == ["RC1", "RC2"]
