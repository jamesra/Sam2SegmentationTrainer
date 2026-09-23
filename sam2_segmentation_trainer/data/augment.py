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
    if gray.ndim == 3 and gray.shape[2] == 3:
        return gray
    if gray.ndim == 3 and gray.shape[2] == 1:
        gray = gray[:, :, 0]
    stacked = np.stack([gray, gray, gray], axis=-1)
    return stacked
