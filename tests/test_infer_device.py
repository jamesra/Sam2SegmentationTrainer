"""EMPredictor(device=None) must resolve the device without a NameError (torch import)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def test_empredictor_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    sam2 = types.ModuleType("sam2")
    sam2_mod = types.ModuleType("sam2.sam2_image_predictor")
    sam2_mod.SAM2ImagePredictor = lambda model: ("predictor", model)  # type: ignore[attr-defined]
    serve = types.ModuleType("sam2_segmentation_trainer.export_serve")
    serve.load_finetuned_model = lambda ckpt, base, cfg, device: ("model", device)  # type: ignore[attr-defined]
    augment = types.ModuleType("sam2_segmentation_trainer.data.augment")
    augment.to_rgb_uint8 = lambda arr: arr  # type: ignore[attr-defined]
    data_pkg = types.ModuleType("sam2_segmentation_trainer.data")
    data_pkg.__path__ = []  # type: ignore[attr-defined]
    for name, mod in (
        ("sam2_segmentation_trainer.data", data_pkg),
        ("sam2_segmentation_trainer.data.augment", augment),
        ("torch", fake_torch),
        ("sam2", sam2),
        ("sam2.sam2_image_predictor", sam2_mod),
        ("sam2_segmentation_trainer.export_serve", serve),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "sam2_segmentation_trainer.infer", raising=False)
    from sam2_segmentation_trainer import infer

    predictor = infer.EMPredictor(Path("ckpt.pt"), base_ckpt=Path("base.pt"))
    assert predictor.device == "cpu"
    monkeypatch.delitem(sys.modules, "sam2_segmentation_trainer.infer", raising=False)
