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
from sam2_segmentation_trainer.data.circle_seed_trust import DEFAULT_OVERSIZE_RANGE, oversize_factor_range
from sam2_segmentation_trainer.data.dataset import (
    FAMILIES_ONLY_PROMPTS,
    VAL_PROMPTS,
    EMSegDataset,
    collate_em,
    limit_main_process_cv2,
    limit_worker_threads,
)
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.prompts import PromptConfig
from sam2_segmentation_trainer.data.splits import include_new_keys, load_split
from sam2_segmentation_trainer.data.tiles import build_tile_context
from sam2_segmentation_trainer.data.valsets import partition_with_gold, resolve_split_path
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
    records: bool = False
    """With ``goal``: also write every per-sample record to ``records.jsonl.gz`` (see :func:`write_records`)."""
    circle_seed_taper: float | None = None
    """With ``goal``, evaluation only: feed circle seeds by the radius-relative taper with this ``full_frac`` (its
    ``zero_frac`` is the resolved diameter scale) instead of the edge confidence; None keeps the model's encoding.
    See docs/circle_seed_trust_eval.md."""
    circle_oversize_range: tuple[float, float] = DEFAULT_OVERSIZE_RANGE
    """With ``goal``: the factor range of the ``circle_oversize`` family's radius over the inscribed circle's."""


RECORDS_FILE = "records.jsonl.gz"
"""Per-sample validation records of a ``--goal --records`` run, one JSON object per line."""


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_records(path: Path, records: dict[str, list[dict]]) -> int:
    """Write ``{family: [record, ...]}`` as gzip JSON lines, one per (family, sample); returns the line count.

    Each line is the record from ``validation.score_sample`` plus ``family``. ``key`` becomes
    ``[volume, image_key, location_id]``. The records keep the raw pixel counts, so ``validation.summarize_mode``
    pools any subset of them exactly as a training run does.
    """
    import gzip

    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for family, rows in records.items():
            for record in rows:
                handle.write(json.dumps({"family": family, **record}, default=_json_default) + "\n")
                count += 1
    tmp.replace(path)
    return count


def read_records(path: Path) -> dict[str, list[dict]]:
    """Read :func:`write_records` output back as ``{family: [record, ...]}`` with ``key`` as a tuple again."""
    import gzip

    records: dict[str, list[dict]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            family = record.pop("family")
            volume, image_key, location_id = record["key"]
            record["key"] = (str(volume), str(image_key), int(location_id))
            records.setdefault(family, []).append(record)
    return records


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
    limit_main_process_cv2()
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=request.num_workers, collate_fn=collate_em,
        worker_init_fn=limit_worker_threads,
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
        if split.val_z:
            # As the trainer does: locations added since the split was written join by their section.
            split = include_new_keys(
                split, examples, train_fraction=float(cfg.data.train_split), seed=int(cfg.training.seed)
            )
    else:
        split, _even = train.make_split_from_cfg(cfg, examples, request.volumes, bool(cfg.data.get("even_per_volume", False)))
        request.split_path.parent.mkdir(parents=True, exist_ok=True)
        save_split(split, request.split_path)
        print(f"wrote split {request.split_path}")
    gold = train.load_gold_set(cfg)
    _train_ex, val_ex = partition_with_gold(gold, *split.partition(examples))
    tiles = build_tile_context(examples, **train._circle_guard(cfg, prompt_cfg))
    approved = train.load_approved_set(cfg, request.data_dir)
    suite = train.build_validation_suite(
        cfg, val_ex, tiles, approved=approved, gold=gold, prompt_cfg=prompt_cfg, num_workers=request.num_workers,
        circle_seed_taper=request.circle_seed_taper, circle_oversize_range=request.circle_oversize_range,
    )
    circle_seed = _circle_seed_options(request, suite)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(request, device)
    print(f"scoring {len(suite.dataset)} validation samples, families {suite.modes}")
    if circle_seed and circle_seed["taper"] is not None:
        print(f"circle seeds fed by the radius-relative taper {circle_seed['taper']} (evaluation only)")
    outcome = suite.run(model, device, request.image_size, request.threshold, full=True, keep_records=request.records)
    results = outcome.full or outcome.subset
    target = suite.core_recall_target
    print(format_summary(0, results, suite.primary, target, label="full", detail=True))
    summary = {
        "model": "zero-shot" if request.finetuned is None else str(request.finetuned),
        "split": str(request.split_path),
        "gold_set": None if gold is None else gold.name,
        "samples": len(suite.dataset),
        "core_band_px": suite.core_band_px,
        "core_recall_target": target,
        # The draw of the per-epoch subset, so a records study can reproduce the run's own subset.
        "subset_seed": suite.seed,
        "epoch_subset": suite.epoch_subset,
        "families": {mode: dataclasses.asdict(result) for mode, result in results.items()},
    }
    if circle_seed:
        summary["circle_seed"] = circle_seed
    request.out_dir.mkdir(parents=True, exist_ok=True)
    (request.out_dir / "goal_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if request.records and outcome.records is not None:
        lines = write_records(request.out_dir / RECORDS_FILE, outcome.records)
        print(f"wrote {lines} per-sample records to {request.out_dir / RECORDS_FILE}")
    return summary


def _circle_seed_options(request: EvalRequest, suite) -> dict:
    """The evaluation-only circle-seed settings of a ``--goal`` run, for ``goal_metrics.json``: the seed taper (None
    when off) and, when ``circle_oversize`` is scored, its factor range. Empty when neither is used, so a default run
    writes the same file as before."""
    taper = suite.dataset.circle_seed_taper
    if taper is None and "circle_oversize" not in suite.modes:
        return {}
    return {
        "taper": None if taper is None else taper.as_dict(),
        "oversize_range": list(request.circle_oversize_range) if "circle_oversize" in suite.modes else None,
    }


def evaluate(request: EvalRequest) -> dict:
    """Score the val split under each prompt. Returns ``{prompt: summary}`` and writes ``metrics.json``."""
    if request.goal:
        return evaluate_goal(request)
    if request.circle_seed_taper is not None:
        raise ValueError("--circle-seed-taper is scored only by --goal (the trainer's validation suite)")
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
        limit_main_process_cv2()
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=request.num_workers,
            collate_fn=collate_em,
            worker_init_fn=limit_worker_threads,
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
        default=None,
        help="split.json to score (use the run's own <run>/split.json), or a frozen split's name (v5, under "
        "{SAM2_CHECKPOINT_ROOT}/valsets/splits); default <out>/splits/zslab_v1.json. With --goal, add "
        "--set data.gold_set=<name> for the gold row",
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
        "--records",
        action="store_true",
        help=f"With --goal: also write every per-sample record to <out>/{RECORDS_FILE} (for "
        "scripts/valset_size_study.py, which measures how large the per-epoch validation subset must be)",
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
    parser.add_argument(
        "--circle-seed-taper",
        type=float,
        default=None,
        metavar="FULL_FRAC",
        help="With --goal, evaluation only: feed every circle-seed family as +seed_logit x w(rho), the radius-relative "
        "cosine taper (full trust inside FULL_FRAC of the true radius, 0 at the diameter scale), instead of the edge "
        "confidence; the disc, negatives and overlap rule are unchanged. Try 0.5. See docs/circle_seed_trust_eval.md",
    )
    parser.add_argument(
        "--circle-oversize-range",
        type=oversize_factor_range,
        default=DEFAULT_OVERSIZE_RANGE,
        metavar="LOW,HIGH",
        help="Factor range of the circle_oversize family's radius over the inscribed circle's (default "
        f"{DEFAULT_OVERSIZE_RANGE[0]},{DEFAULT_OVERSIZE_RANGE[1]}); score it with --goal --set "
        "eval.val_prompts=[circle_mask,circle_annotator,circle_oversize]",
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
    split_path = resolve_split_path(args.split) if args.split else (out_root / "splits" / "zslab_v1.json")
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
            records=args.records,
            circle_seed_taper=args.circle_seed_taper,
            circle_oversize_range=args.circle_oversize_range,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
