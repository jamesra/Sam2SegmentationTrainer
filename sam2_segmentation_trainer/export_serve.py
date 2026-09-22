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


def export_serve_checkpoint(
    *,
    finetuned: Path,
    base_ckpt: Path,
    out: Path,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cpu",
) -> Path:
    """Load base SAM2, overlay fine-tuned weights, write a build_sam2 checkpoint."""
    import torch

    from sam2_segmentation_trainer.model import build_em_sam2

    payload = torch.load(finetuned, map_location="cpu", weights_only=True)
    state = finetuned_state_dict(payload)
    model = build_em_sam2(model_cfg, str(base_ckpt), torch.device(device), train=False)
    model.load_state_dict(state)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
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
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
