"""Evaluate a SAM2 checkpoint on the val split, once per validation prompt."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
from sam2_segmentation_trainer.data.dataset import FAMILIES_ONLY_PROMPTS, VAL_PROMPTS, EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import PromptConfig
from sam2_segmentation_trainer.data.splits import load_split
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.encoding_record import ResolvedEncoding, read_record, resolve_encoding
from sam2_segmentation_trainer.export_serve import load_finetuned_model
from sam2_segmentation_trainer.interactive import simulate_clicks, summarize_noc
from sam2_segmentation_trainer.metrics import boundary_f1, iou_dice_pr, summarize_records
from sam2_segmentation_trainer.overlays import denorm_preview, overlay_panel
from sam2_segmentation_trainer.paths import data_root, output_root, pretrained_checkpoint
from sam2_segmentation_trainer.validation import DEFAULT_VAL_PROMPTS, predict_batch

PER_PROMPT_DEFAULTS = tuple(p for p in DEFAULT_VAL_PROMPTS if p not in FAMILIES_ONLY_PROMPTS)
"""Prompts the one-prompt-at-a-time evaluation scores by default. Families that only the validation suite builds
(``circle_annotator``) are left out; ``--goal`` scores them."""


@dataclass(frozen=True, slots=True)
class EvalRequest:
    """Arguments for one evaluation run.

    ``finetuned`` None is the zero-shot baseline: the base checkpoint with no fine-tuned weights.
    """

    finetuned: Path | None
    base_ckpt: Path
    model_cfg: str
    data_dir: Path
    volumes: list[str]
    split_path: Path
    image_size: int
    threshold: float
    out_dir: Path
    grid_every: int
    max_grids: int
    num_workers: int
    prompts: tuple[str, ...] = PER_PROMPT_DEFAULTS
    noc: bool = False
    noc_max_clicks: int = 20
    noc_target_iou: float = 0.9
    circle_scale: float | None = None
    """Override of the diameter scale of seed circles; None uses the model's record, else the default."""
    circle_full_frac: float | None = None
    circle_zero_frac: float | None = None
    """Overrides of ``prompt.circle_annotation_trust``; None uses the model's record, else the default."""
    core_band_px: float = 10.0
    """``val.core_band_px``: the band the goal metrics and the NoC click sampling ignore around the traced edge."""
    core_recall_target: float = 0.975
    merge_min_px: int = 500
    goal: bool = False
    """Score the full validation split with the training validation suite (every family's goal metrics)."""
    overrides: tuple[str, ...] = ()
    """Config overrides (``section.key=value``) for the ``goal`` run, like the trainer's command line."""


def resolve_request_encoding(request: EvalRequest) -> ResolvedEncoding:
    """The prompt encoding to score ``request`` with: the model's own record when it has one, else the defaults,
    with the command-line overrides applied and logged."""
    record, source = read_record(request.finetuned)
    return resolve_encoding(
        record,
        source=source,
        circle_scale=request.circle_scale,
        full_frac=request.circle_full_frac,
        zero_frac=request.circle_zero_frac,
    )


def _load_model(request: EvalRequest, device: torch.device):
    if request.finetuned is not None:
        return load_finetuned_model(request.finetuned, request.base_ckpt, request.model_cfg, str(device))
    from sam2_segmentation_trainer.model import build_em_sam2

    model = build_em_sam2(request.model_cfg, str(request.base_ckpt), device, train=False)
    model.eval()
    return model


def _evaluate_noc(request: EvalRequest, val_ex, tiles, model, device, prompt_cfg=None) -> dict:
    """Clicks to reach `noc_target_iou` from one interior point plus simulated corrections, per volume."""
    dataset = EMSegDataset(
        val_ex, split="val", image_size=request.image_size, tiles=tiles, val_prompt="point",
        core_band_px=request.core_band_px, prompt_cfg=prompt_cfg,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=request.num_workers, collate_fn=collate_em
    )
    rows: list[dict] = []
    volumes: list[str] = []
    for i, batch in enumerate(tqdm(loader, desc="eval noc")):
        for row in simulate_clicks(
            model,
            batch,
            device,
            request.image_size,
            max_clicks=request.noc_max_clicks,
            target_iou=request.noc_target_iou,
            threshold=request.threshold,
            seed=i,
            core_band_px=request.core_band_px,
            core_recall_target=request.core_recall_target,
            merge_min_px=request.merge_min_px,
        ):
            rows.append(row)
            volumes.append(str(batch["meta"][0]["volume"]))
    summary = summarize_noc(rows, volumes, request.noc_max_clicks)
    summary["target_iou"] = request.noc_target_iou
    summary["core_recall_target"] = request.core_recall_target
    return summary


def evaluate_goal(request: EvalRequest) -> dict:
    """One full validation pass with the trainer's own suite: the goal metrics of every prompt family.

    This is what a training run logs as ``val_full``, so the zero-shot model and earlier checkpoints can be scored
    the same way as a new run. The split is ``request.split_path`` (it is created from the config when missing, so
    pass the new run's own ``split.json`` to get its validation set).
    """
    import dataclasses

    from sam2_segmentation_trainer import train
    from sam2_segmentation_trainer.data.manifest import take_model_sized
    from sam2_segmentation_trainer.data.splits import save_split
    from sam2_segmentation_trainer.validation import format_summary

    resolved = resolve_request_encoding(request)
    # The model's record and the command line come first, so an explicit --set still wins.
    cfg = train.load_train_cfg(resolved.cfg_overrides() + list(request.overrides))
    prompt_cfg = PromptConfig.from_cfg(cfg)
    examples = index_volumes(volumes=request.volumes, root=request.data_dir, skip_missing=True)
    examples, _wrong_size = take_model_sized(examples, request.image_size)
    if request.split_path.is_file():
        split = load_split(request.split_path)
    else:
        split, _even = train.make_split_from_cfg(cfg, examples, request.volumes, bool(cfg.data.get("even_per_volume", False)))
        request.split_path.parent.mkdir(parents=True, exist_ok=True)
        save_split(split, request.split_path)
        print(f"wrote split {request.split_path}")
    _train_ex, val_ex = split.partition(examples)
    tiles = build_tile_context(examples, **train._circle_guard(cfg, prompt_cfg))
    approved = train.load_approved_set(cfg, request.data_dir)
    suite = train.build_validation_suite(
        cfg, val_ex, tiles, approved=approved, prompt_cfg=prompt_cfg, num_workers=request.num_workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(request, device)
    print(f"scoring {len(suite.dataset)} validation samples, families {suite.modes}")
    outcome = suite.run(model, device, request.image_size, request.threshold, full=True)
    results = outcome.full or outcome.subset
    target = suite.core_recall_target
    print(format_summary(0, results, suite.primary, target, label="full", detail=True))
    summary = {
        "model": "zero-shot" if request.finetuned is None else str(request.finetuned),
        "split": str(request.split_path),
        "samples": len(suite.dataset),
        "core_band_px": suite.core_band_px,
        "core_recall_target": target,
        "families": {mode: dataclasses.asdict(result) for mode, result in results.items()},
    }
    request.out_dir.mkdir(parents=True, exist_ok=True)
    (request.out_dir / "goal_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate(request: EvalRequest) -> dict:
    """Score the val split under each prompt. Returns ``{prompt: summary}`` and writes ``metrics.json``."""
    if request.goal:
        return evaluate_goal(request)
    unknown = [p for p in request.prompts if p not in VAL_PROMPTS]
    if unknown:
        raise ValueError(f"unknown prompt(s) {unknown}; choose from {list(VAL_PROMPTS)}")
    families_only = [p for p in request.prompts if p in FAMILIES_ONLY_PROMPTS]
    if families_only:
        raise ValueError(f"{families_only} are scored only by --goal (the trainer's validation suite)")
    examples = index_volumes(volumes=request.volumes, root=request.data_dir, skip_missing=True)
    split = load_split(request.split_path)
    _train_ex, val_ex = split.partition(examples)
    resolved = resolve_request_encoding(request)
    prompt_cfg = PromptConfig(**resolved.prompt_kwargs())
    tiles = build_tile_context(examples, circle_scale=resolved.circle_diameter_scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(request, device)

    summaries: dict[str, dict] = {}
    grid_dir = request.out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for prompt in request.prompts:
        dataset = EMSegDataset(
            val_ex, split="val", image_size=request.image_size,
            tiles=tiles,
            val_prompt=prompt,
            circle_scale=resolved.circle_diameter_scale,
            prompt_cfg=prompt_cfg,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=request.num_workers,
            collate_fn=collate_em,
        )
        records = []
        grids_written = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc=f"eval {prompt}")):
                pred = predict_batch(model, batch, device, request.image_size)
                pred_bin = (torch.sigmoid(pred) > request.threshold)[0].cpu().numpy().astype(np.uint8)
                gt = batch["mask"][0].cpu().numpy().astype(np.uint8)
                stats = iou_dice_pr(pred_bin, gt)
                stats["bf"] = boundary_f1(pred_bin, gt)
                meta = batch["meta"][0]
                stats.update(meta)
                records.append(stats)
                if request.grid_every > 0 and i % request.grid_every == 0 and grids_written < request.max_grids:
                    panel = overlay_panel(denorm_preview(batch["image"][0]), gt, pred_bin)
                    name = f"{prompt}_{meta['volume']}_{meta['image_key']}_{meta['location_id']}.png"
                    Image.fromarray(panel).save(grid_dir / name)
                    grids_written += 1
        summaries[prompt] = summarize_records(records)

    if request.noc:
        summaries["noc"] = _evaluate_noc(request, val_ex, tiles, model, device, prompt_cfg)

    request.out_dir.mkdir(parents=True, exist_ok=True)
    (request.out_dir / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on the val split, once per prompt")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fine-tuned state_dict")
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Score the base checkpoint with no fine-tuned weights (the baseline to beat); replaces --checkpoint",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--volumes", default="RC1,RC2,RPC1,RPC2")
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="split.json to score (use the run's own <run>/split.json); default <out>/splits/zslab_v1.json",
    )
    parser.add_argument(
        "--prompts",
        default=",".join(PER_PROMPT_DEFAULTS),
        help=f"Comma-separated validation prompts from {list(VAL_PROMPTS)}",
    )
    parser.add_argument(
        "--noc",
        action="store_true",
        help="Also report NoC: clicks needed to reach --noc-iou with simulated corrective clicks",
    )
    parser.add_argument(
        "--goal",
        action="store_true",
        help="Score the FULL validation split with the trainer's validation suite and print every prompt family's "
        "goal metrics (core recall, merge, spill, ...); writes goal_metrics.json. Needs --split (created from the "
        "config when the file is missing, so the new run's own split.json can be given before the run starts).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override for --goal, e.g. --set data.split_mode=zslab (repeatable)",
    )
    parser.add_argument("--noc-iou", type=float, default=0.9)
    parser.add_argument("--noc-max-clicks", type=int, default=20)
    parser.add_argument(
        "--core-band-px", type=float, default=10.0,
        help="val.core_band_px: simulated clicks target errors more than this far from the traced edge",
    )
    parser.add_argument(
        "--core-recall-target", type=float, default=0.975,
        help="Second NoC stopping rule: core recall at least this with no merge",
    )
    parser.add_argument(
        "--circle-scale",
        type=float,
        default=None,
        help="Override the diameter scale of the seed circles (default: the model's prompt-encoding record, else "
        f"{DEFAULT_CIRCLE_DIAMETER_SCALE}). The exporter writes true radii; this applies to seed circles only",
    )
    parser.add_argument(
        "--circle-full-frac",
        type=float,
        default=None,
        help="Override prompt.circle_annotation_trust.full_frac (default: the model's record, else the library default)",
    )
    parser.add_argument(
        "--circle-zero-frac",
        type=float,
        default=None,
        help="Override prompt.circle_annotation_trust.zero_frac (default: the model's record, else the library default)",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--grid-every", type=int, default=10)
    parser.add_argument("--max-grids", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.zero_shot == (args.checkpoint is not None):
        parser.error("give exactly one of --checkpoint and --zero-shot")

    data_dir = data_root(args.root)
    out_root = output_root()
    split_path = args.split or (out_root / "splits" / "zslab_v1.json")
    out_dir = args.out or (out_root / ("eval_zero_shot" if args.zero_shot else "eval"))
    base = args.base_checkpoint or pretrained_checkpoint(out_root)
    volumes = [v.strip() for v in args.volumes.split(",") if v.strip()]
    prompts = tuple(p.strip() for p in args.prompts.split(",") if p.strip())
    summary = evaluate(
        EvalRequest(
            finetuned=None if args.zero_shot else args.checkpoint,
            base_ckpt=base,
            model_cfg=args.model_config,
            data_dir=data_dir,
            volumes=volumes,
            split_path=split_path,
            image_size=args.image_size,
            threshold=args.threshold,
            out_dir=out_dir,
            grid_every=args.grid_every,
            max_grids=args.max_grids,
            num_workers=args.num_workers,
            prompts=prompts,
            noc=args.noc,
            noc_max_clicks=args.noc_max_clicks,
            noc_target_iou=args.noc_iou,
            circle_scale=args.circle_scale,
            circle_full_frac=args.circle_full_frac,
            circle_zero_frac=args.circle_zero_frac,
            core_band_px=args.core_band_px,
            core_recall_target=args.core_recall_target,
            goal=args.goal,
            overrides=tuple(args.overrides),
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
