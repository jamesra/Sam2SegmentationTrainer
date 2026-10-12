"""EM Albumentations: geometry and grayscale-safe intensity, no hue/saturation."""

from __future__ import annotations

import albumentations as A
import numpy as np


def build_train_transform() -> A.Compose:
    """Geometry and intensity only. Tiles are already the model size, so this does not resample."""
    # GaussNoise API is std_range as a fraction of 255 (replaces var_limit).
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(std_range=(0.04, 0.20), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.2, p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Blur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        ]
    )


def build_val_transform() -> A.Compose:
    return A.Compose([])


def to_rgb_uint8(gray: np.ndarray) -> np.ndarray:
    """Stack a gray plane to HxWx3 uint8. An array that is already uint8 is not rescaled."""
    array = np.asarray(gray)
    if array.ndim == 3 and array.shape[2] == 3:
        channels = array
    elif array.ndim == 3 and array.shape[2] == 1:
        plane = array[:, :, 0]
        channels = np.stack([plane, plane, plane], axis=-1)
    else:
        channels = np.stack([array, array, array], axis=-1)
    if channels.dtype == np.uint8:
        return np.ascontiguousarray(channels)
    return np.ascontiguousarray(channels, dtype=np.uint8)
