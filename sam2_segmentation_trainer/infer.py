"""Drop-in SAM2ImagePredictor wrapper using fine-tuned weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam2_segmentation_trainer.model import build_em_sam2
from sam2_segmentation_trainer.paths import pretrained_checkpoint


def normalise_em(arr: np.ndarray) -> np.ndarray:
    """Percentile stretch robust to hot/dead pixels, then 8-bit."""
    lo, hi = np.percentile(arr, [1, 99])
    clipped = np.clip(arr.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)


def load_em_rgb(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as img:
        arr = np.array(img.convert("L"))
    stretched = normalise_em(arr)
    return np.stack([stretched, stretched, stretched], axis=-1)


class EMPredictor:
    """SAM2ImagePredictor with a fine-tuned SAM2 state_dict."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        base_ckpt: str | Path | None = None,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        base = Path(base_ckpt) if base_ckpt is not None else pretrained_checkpoint()
        model = build_em_sam2(model_cfg, str(base), torch.device(device), train=False)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image_path: str | Path) -> None:
        rgb = load_em_rgb(image_path)
        self.predictor.set_image(rgb)

    def set_image_array(self, gray_or_rgb: np.ndarray) -> None:
        if gray_or_rgb.ndim == 2:
            stretched = normalise_em(gray_or_rgb)
            rgb = np.stack([stretched, stretched, stretched], axis=-1)
        else:
            rgb = gray_or_rgb
        self.predictor.set_image(rgb)

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask: bool = True,
    ):
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EM SAM2 inference on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--point", default=None, help="x,y foreground click in pixels")
    parser.add_argument("--out", type=Path, default=Path("output_mask.png"))
    args = parser.parse_args(argv)

    predictor = EMPredictor(
        checkpoint=args.checkpoint,
        model_cfg=args.model_config,
        base_ckpt=args.base_checkpoint,
    )
    predictor.set_image(args.image)
    if args.point:
        x_str, y_str = args.point.split(",")
        coords = np.array([[float(x_str), float(y_str)]])
        labels = np.array([1])
        masks, scores, _logits = predictor.predict(
            point_coords=coords, point_labels=labels, multimask=True
        )
    else:
        raise SystemExit("provide --point x,y")
    best = masks[int(np.argmax(scores))]
    Image.fromarray((best.astype(np.uint8)) * 255).save(args.out)
    print(f"saved {args.out} (score {float(np.max(scores)):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
