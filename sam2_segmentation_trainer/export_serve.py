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


def run_encoding_settings(run_dir: Path | None) -> dict[str, float] | None:
    """The encoding values a run trained with, from its saved ``config.yaml``; None when there is none to read.

    Keys: ``edge_tau_px``, ``edge_tau_depth_frac``, ``seed_logit``, ``correction_logit``,
    ``circle_diameter_scale`` (the ``prompt.*`` values; a key the config lacks keeps the library default).
    """
    from omegaconf import OmegaConf

    from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
    from annotation_crops import prompt_encoding as enc

    path = None if run_dir is None else Path(run_dir) / "config.yaml"
    if path is None or not path.is_file():
        return None
    try:
        cfg = OmegaConf.load(path)
    except Exception as exc:  # unreadable yaml: treat as missing, the caller warns
        print(f"WARNING: could not read {path}: {exc}")
        return None
    defaults = {
        "edge_tau_px": enc.EDGE_TAU_PX,
        "edge_tau_depth_frac": enc.EDGE_TAU_DEPTH_FRAC,
        "seed_logit": enc.SEED_LOGIT,
        "correction_logit": enc.CORRECTION_LOGIT,
        "circle_diameter_scale": DEFAULT_CIRCLE_DIAMETER_SCALE,
    }
    return {
        key: float(OmegaConf.select(cfg, f"prompt.{key}", default=default)) for key, default in defaults.items()
    }


def serve_metadata(circle_diameter_scale: float | None = None, run_dir: Path | None = None) -> dict[str, Any]:
    """The prompt-encoding constants written next to the weights (``"prompt_encoding"`` in the exported file).

    The values come from the run's own saved config (``run_dir/config.yaml``). Without one, the library defaults are
    used and a warning is printed. ``circle_diameter_scale`` overrides the run's value. The mask network was trained
    on ``mask_input`` built by ``annotation_crops/prompt_encoding.py`` with these constants; a server must build it
    the same way (see the README), and a run whose values differ from the library constants is warned about.
    """
    from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
    from annotation_crops.prompt_encoding import encoding_differences, encoding_metadata

    settings = run_encoding_settings(run_dir)
    if settings is None:
        print(
            "WARNING: no saved config.yaml for this run; writing the library default encoding constants. "
            "Give --run-dir if the run used non-default prompt.* values."
        )
        settings = {}
    if circle_diameter_scale is not None:
        settings["circle_diameter_scale"] = float(circle_diameter_scale)
    scale = settings.pop("circle_diameter_scale", DEFAULT_CIRCLE_DIAMETER_SCALE)
    meta = encoding_metadata(scale, **settings)
    for message in encoding_differences(
        edge_tau_px=meta["EDGE_TAU_PX"],
        edge_tau_depth_frac=meta["EDGE_TAU_DEPTH_FRAC"],
        seed_logit=meta["SEED_LOGIT"],
        correction_logit=meta["CORRECTION_LOGIT"],
        circle_diameter_scale=scale,
        default_circle_diameter_scale=DEFAULT_CIRCLE_DIAMETER_SCALE,
    ):
        print(f"WARNING: {message}; the server implements the library constants")
    return meta


def export_serve_checkpoint(
    *,
    finetuned: Path,
    base_ckpt: Path,
    out: Path,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cpu",
    circle_diameter_scale: float | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Load base SAM2, overlay fine-tuned weights, write a build_sam2 checkpoint.

    The file is ``{"model": state_dict, "prompt_encoding": {...}}``: ``build_sam2`` reads only ``"model"``, and the
    metadata records the constants the model was trained with, read from ``run_dir/config.yaml`` (the checkpoint's
    own folder when ``run_dir`` is omitted; see :func:`serve_metadata`).
    """
    import torch

    model = load_finetuned_model(finetuned, base_ckpt, model_cfg, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(finetuned).parent if run_dir is None else Path(run_dir)
    metadata = serve_metadata(circle_diameter_scale, run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "prompt_encoding": metadata}, out)
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
        help="Override the run's prompt.circle_diameter_scale recorded in the file",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run folder holding the saved config.yaml (default: the folder of --checkpoint)",
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
        run_dir=args.run_dir,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
