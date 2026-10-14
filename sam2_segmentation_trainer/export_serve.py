"""Export a trainer checkpoint into the Meta dict that build_sam2 loads."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def finetuned_state_dict(payload: object) -> dict[str, Any]:
    """Parameter tensors from a raw state_dict or a ``{"model": ...}`` wrapper.

    Trainer ``best_model.pt`` is a raw ``state_dict``. SegmentationServer's
    ``build_sam2`` expects the Meta layout ``{"model": state_dict}``.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    nested = payload.get("model")
    if isinstance(nested, dict) and not _looks_like_state_dict(payload):
        return nested
    if _looks_like_state_dict(payload):
        return payload
    raise ValueError("checkpoint has neither a model state_dict nor a 'model' wrapper")


def _looks_like_state_dict(payload: dict[str, Any]) -> bool:
    return any(isinstance(key, str) and key.startswith("image_encoder") for key in payload)


def load_finetuned_model(
    finetuned: Path,
    base_ckpt: Path,
    model_cfg: str,
    device: str,
):
    """Eval-mode SAM2 with fine-tuned weights from a raw state dict or ``{"model": ...}``."""
    import torch

    from sam2_segmentation_trainer.model import build_em_sam2

    torch_device = torch.device(device)
    payload = torch.load(finetuned, map_location=torch_device, weights_only=True)
    state = finetuned_state_dict(payload)
    model = build_em_sam2(model_cfg, str(base_ckpt), torch_device, train=False)
    model.load_state_dict(state)
    model.eval()
    return model


def serve_metadata(circle_diameter_scale: float | None = None) -> dict[str, Any]:
    """The prompt-encoding constants written next to the weights (``"prompt_encoding"`` in the exported file).

    The mask network of a v4 model was trained on ``mask_input`` built by ``annotation_crops/prompt_encoding.py``
    with these constants; a server must build it the same way (see the README).
    """
    from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
    from annotation_crops.prompt_encoding import encoding_metadata

    scale = DEFAULT_CIRCLE_DIAMETER_SCALE if circle_diameter_scale is None else circle_diameter_scale
    return encoding_metadata(scale)


def export_serve_checkpoint(
    *,
    finetuned: Path,
    base_ckpt: Path,
    out: Path,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cpu",
    circle_diameter_scale: float | None = None,
) -> Path:
    """Load base SAM2, overlay fine-tuned weights, write a build_sam2 checkpoint.

    The file is ``{"model": state_dict, "prompt_encoding": {...}}``: ``build_sam2`` reads only ``"model"``, and the
    metadata records the constants the model was trained with (:func:`serve_metadata`).
    """
    import torch

    model = load_finetuned_model(finetuned, base_ckpt, model_cfg, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "prompt_encoding": serve_metadata(circle_diameter_scale)}, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a Meta-format SAM2 checkpoint from a trainer state_dict"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trainer best_model.pt (raw state_dict) or an existing {model: ...} file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination, e.g. best_TEM_model.pt",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--circle-diameter-scale",
        type=float,
        default=None,
        help="prompt.circle_diameter_scale of the training run, recorded in the file (default: the library default)",
    )
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only to build the base model (default cpu; does not change weights)",
    )
    args = parser.parse_args(argv)
    from sam2_segmentation_trainer.paths import pretrained_checkpoint

    base = args.base_checkpoint or pretrained_checkpoint()
    out = export_serve_checkpoint(
        finetuned=args.checkpoint,
        base_ckpt=base,
        out=args.out,
        model_cfg=args.model_config,
        device=args.device,
        circle_diameter_scale=args.circle_diameter_scale,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
