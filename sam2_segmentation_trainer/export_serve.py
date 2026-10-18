"""Export a trainer checkpoint into the Meta dict that build_sam2 loads."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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


def run_encoding_settings(run_dir: Path | None) -> dict[str, Any] | None:
    """The encoding values a run trained with, from its saved ``config.yaml``; None when there is none to read.

    Keys: ``edge_tau_px``, ``edge_tau_depth_frac``, ``seed_logit``, ``correction_logit``,
    ``circle_diameter_scale`` (floats) and ``circle_annotation_trust`` (a ``CircleAnnotationTrust``): the
    ``prompt.*`` values; a key the config lacks keeps the library default.
    """
    from omegaconf import OmegaConf

    from sam2_segmentation_trainer.encoding_record import settings_from_cfg

    path = None if run_dir is None else Path(run_dir) / "config.yaml"
    if path is None or not path.is_file():
        return None
    try:
        cfg = OmegaConf.load(path)
    except Exception as exc:  # unreadable yaml: treat as missing, the caller warns
        print(f"WARNING: could not read {path}: {exc}")
        return None
    return settings_from_cfg(cfg)


def serve_metadata(circle_diameter_scale: float | None = None, run_dir: Path | None = None) -> dict[str, Any]:
    """The prompt-encoding constants written next to the weights (``"prompt_encoding"`` in the exported file).

    The values come from the run's own saved config (``run_dir/config.yaml``). Without one, the library defaults are
    used and a warning is printed. ``circle_diameter_scale`` overrides the run's value. The mask network was trained
    on ``mask_input`` built by ``annotation_crops/prompt_encoding.py`` with these constants; a server must build it
    the same way (see the README), and a run whose values differ from the library constants is warned about.
    """
    from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
    from annotation_crops.prompt_encoding import CircleAnnotationTrust, encoding_differences, encoding_metadata

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
        circle_annotation_trust=CircleAnnotationTrust.from_mapping(meta["circle_annotation_trust"]),
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
    model_id: str | None = None,
    epoch: int | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> Path:
    """Load base SAM2, overlay fine-tuned weights, write a build_sam2 checkpoint.

    The file is ``{"model": state_dict, "prompt_encoding": {...}}``: ``build_sam2`` reads only ``"model"``, and the
    metadata records the constants the model was trained with, read from ``run_dir/config.yaml`` (the checkpoint's
    own folder when ``run_dir`` is omitted; see :func:`serve_metadata`). The same dict is also written as
    ``<out stem>.prompt_encoding.json`` next to ``out``. The exporter never scales circles: the record carries the
    diameter scale and the circle-annotation trust fractions for the server to apply.

    A model manifest (``<out stem>.model.json``, see ``model_manifest.py``) is written beside the file too: the model
    id, its version, the run's training-data record and the encoding. The model is named ``v<VERSION>.<EPOCH>``, and
    the file name may lead with labels joined by ``-`` (``sem-v1.5.pt``); a file name that ends in a different
    version is refused (``.cursor/rules/naming-conventions.mdc``): ``v<VERSION>`` comes from the run name (``sam2_em_v4`` is ``v4``) unless
    ``model_id`` says otherwise, and ``epoch`` is the trainer's epoch index of these weights (read from
    ``best_model.json`` beside a ``best_model.pt`` when omitted).

    ``description`` is the operator's text about the model. When omitted, a description already in the manifest at
    ``out`` is kept, so re-exporting never erases what someone wrote.

    Raises:
        ValueError: The model version or the epoch cannot be told, or ``out`` is named for another model version.
            Checked before any weights are loaded.
    """
    import torch

    from sam2_segmentation_trainer.data_record import UNKNOWN_DATA, read_data_record
    from sam2_segmentation_trainer.encoding_record import sidecar_path, write_record
    from sam2_segmentation_trainer.model_manifest import (
        MODEL_ID_RE,
        build_manifest,
        manifest_path,
        model_id_for_run,
        model_version_name,
        stem_mismatch,
        write_manifest,
    )

    run_dir = Path(finetuned).parent if run_dir is None else Path(run_dir)
    run_name = _run_name(run_dir)
    resolved_id = model_id or model_id_for_run(run_name)
    if resolved_id is None or not MODEL_ID_RE.match(resolved_id):
        raise ValueError(
            f"cannot tell the model version from run {run_name!r} / model id {model_id!r}: "
            f"pass --model-id v<VERSION>, for example v4"
        )
    if epoch is None:
        epoch = best_epoch(Path(finetuned))
    if epoch is None:
        raise ValueError(
            "the epoch these weights come from is not known: pass --epoch "
            "(a best_model.pt export reads best_model.json beside it, which runs started after the naming convention write)"
        )
    # A file named for a model must be named for this one: v1.6.pt or sem-v1.6.pt for epoch 6, not epoch 5.
    mismatch = stem_mismatch(Path(out).stem, model_version_name(resolved_id, epoch))
    if mismatch:
        raise ValueError(mismatch)

    model = load_finetuned_model(finetuned, base_ckpt, model_cfg, device)
    metadata = serve_metadata(circle_diameter_scale, run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "prompt_encoding": metadata}, out)
    write_record(sidecar_path(out), metadata)
    sidecar = manifest_path(out)
    write_manifest(
        sidecar,
        build_manifest(
            model_id=resolved_id,
            model_version=model_version_name(resolved_id, epoch),
            encoding=metadata,
            training_data=read_data_record(run_dir) or UNKNOWN_DATA,
            run=run_name,
            epoch=epoch,
            display_name=display_name,
            description=description if description is not None else kept_description(sidecar),
            base_checkpoint=Path(base_ckpt).name,
        ),
    )
    return out


BEST_EPOCH_FILE = "best_model.json"


def best_epoch(weights: Path) -> int | None:
    """The epoch of a ``best_model.pt``, from ``best_model.json`` beside it; None for any other file or no record."""
    import json

    if weights.name != "best_model.pt":
        return None
    try:
        record = json.loads((weights.parent / BEST_EPOCH_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    epoch = record.get("epoch") if isinstance(record, dict) else None
    return epoch if isinstance(epoch, int) and epoch >= 0 else None


def kept_description(manifest_file: Path) -> str:
    """The operator's description already in the manifest at ``manifest_file``, or an empty string."""
    import json

    try:
        existing = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
        text = existing["model"]["description"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return ""
    return text if isinstance(text, str) else ""


def _run_name(run_dir: Path) -> str:
    """The run's ``training.run_name`` from its saved ``config.yaml``, else the folder name."""
    from omegaconf import OmegaConf

    path = Path(run_dir) / "config.yaml"
    if path.is_file():
        try:
            name = OmegaConf.select(OmegaConf.load(path), "training.run_name", default=None)
        except Exception:
            name = None
        if name:
            return str(name)
    return Path(run_dir).name


SERVE_DIR = "serve"


def launch_epoch_export(model: Any, run_dir: Path, epoch: int) -> Path:
    """Start a separate process that writes the serve file for ``model``'s current weights; return its path.

    The weights go to ``<run_dir>/serve/serve_epoch<N>_weights.pt`` first (the child cannot see this process's
    memory), then ``python -m sam2_segmentation_trainer.export_serve`` runs detached on the CPU and writes
    ``serve_epoch<N>.pt`` plus its ``.prompt_encoding.json`` and ``.model.json`` beside it, then deletes the weights
    file. Training does not wait for it. Its output goes to ``serve/export.log``.
    """
    import torch

    serve = Path(run_dir) / SERVE_DIR
    serve.mkdir(parents=True, exist_ok=True)
    weights = serve / f"serve_epoch{epoch}_weights.pt"
    out = serve / f"serve_epoch{epoch}.pt"
    torch.save(model.state_dict(), weights)  # only the child reads it, after this returns
    command = [
        sys.executable, "-m", "sam2_segmentation_trainer.export_serve",
        "--checkpoint", str(weights), "--out", str(out), "--run-dir", str(run_dir), "--epoch", str(epoch),
        "--delete-checkpoint",
    ]
    detach: dict[str, Any] = (
        {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with open(serve / "export.log", "ab") as log:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # the export builds the model on the CPU
            **detach,
        )
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
        "--model-id", default=None, help="Model id v<VERSION> (default: from the run name, sam2_em_v4 is v4)"
    )
    parser.add_argument("--display-name", default=None, help="Human-readable model name for the manifest")
    parser.add_argument(
        "--description",
        default=None,
        help="What this model is, for the manifest. Omit to keep the description already in the manifest at --out",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Epoch the weights come from, for the model version")
    parser.add_argument(
        "--delete-checkpoint",
        action="store_true",
        help="Delete --checkpoint after the export succeeds (the trainer's per-epoch weights snapshot)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only to build the base model (default cpu; does not change weights)",
    )
    args = parser.parse_args(argv)
    from sam2_segmentation_trainer.paths import pretrained_checkpoint

    base = args.base_checkpoint or pretrained_checkpoint()
    try:
        out = _export_from_args(args, base)
    except ValueError as exc:
        parser.error(str(exc))
    print(out)
    return 0


def _export_from_args(args: argparse.Namespace, base: Path) -> Path:
    out = export_serve_checkpoint(
        finetuned=args.checkpoint,
        base_ckpt=base,
        out=args.out,
        model_cfg=args.model_config,
        device=args.device,
        circle_diameter_scale=args.circle_diameter_scale,
        run_dir=args.run_dir,
        model_id=args.model_id,
        epoch=args.epoch,
        display_name=args.display_name,
        description=args.description,
    )
    if args.delete_checkpoint:
        args.checkpoint.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
