"""data.elastic_warp: the training transform warps only when asked; validation never augments."""

from __future__ import annotations

import albumentations as A
from omegaconf import OmegaConf

from sam2_segmentation_trainer.data.augment import build_train_transform, build_val_transform


def _has_elastic(transform: A.Compose) -> bool:
    return any(isinstance(t, A.ElasticTransform) for t in transform.transforms)


def test_elastic_warp_is_off_by_default():
    assert not _has_elastic(build_train_transform())


def test_elastic_warp_can_be_turned_on():
    assert _has_elastic(build_train_transform(elastic_warp=True))


def test_validation_never_augments():
    assert len(build_val_transform().transforms) == 0


def test_config_default_is_off():
    from pathlib import Path

    import sam2_segmentation_trainer

    path = Path(sam2_segmentation_trainer.__file__).parent / "config" / "em_finetune.yaml"
    cfg = OmegaConf.load(path)
    assert OmegaConf.select(cfg, "data.elastic_warp") is False
