"""Evaluate a fine-tuned SAM2 checkpoint on the val split."""

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

from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.splits import load_split
from sam2_segmentation_trainer.export_serve import load_finetuned_model
from sam2_segmentation_trainer.metrics import boundary_f1, iou_dice_pr, summarize_records
from sam2_segmentation_trainer.model import predict_masks
from sam2_segmentation_trainer.overlays import denorm_preview, overlay_panel
from sam2_segmentation_trainer.paths import data_root, output_root, pretrained_checkpoint


@dataclass(frozen=True, slots=True)
class EvalRequest:
    """Arguments for one evaluation run."""

    finetuned: Path
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


def evaluate(request: EvalRequest) -> dict:
    examples = index_volumes(volumes=request.volumes, root=request.data_dir, skip_missing=True)
    split = load_split(request.split_path)
    _train_ex, val_ex = split.partition(examples)
    dataset = EMSegDataset(val_ex, split="val", image_size=request.image_size)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=request.num_workers,
        collate_fn=collate_em,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned_model(
        request.finetuned,
        request.base_ckpt,
        request.model_cfg,
        str(device),
    )

    records = []
    grid_dir = request.out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grids_written = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="eval")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            points = batch["point"].to(device)
            labels = batch["point_label"].to(device)
            pred, _iou = predict_masks(
                model, images, points, labels, image_size=request.image_size
            )
            pred_bin = (torch.sigmoid(pred) > request.threshold)[0].cpu().numpy().astype(np.uint8)
            gt = masks[0].cpu().numpy().astype(np.uint8)
            stats = iou_dice_pr(pred_bin, gt)
            stats["bf"] = boundary_f1(pred_bin, gt)
            meta = batch["meta"][0]
            stats.update(meta)
            records.append(stats)
            if request.grid_every > 0 and i % request.grid_every == 0 and grids_written < request.max_grids:
                preview = denorm_preview(batch["image"][0])
                panel = overlay_panel(preview, gt, pred_bin)
                name = f"{meta['volume']}_{meta['image_key']}_{meta['location_id']}.png"
                Image.fromarray(panel).save(grid_dir / name)
                grids_written += 1

    summary = summarize_records(records)
    request.out_dir.mkdir(parents=True, exist_ok=True)
    (request.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SAM2 on the val split")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned state_dict")
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--volumes", default="RC1,RC2,RPC1,RPC2")
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--grid-every", type=int, default=10)
    parser.add_argument("--max-grids", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    data_dir = data_root(args.root)
    out_root = output_root()
    split_path = args.split or (out_root / "splits" / "location_v1.json")
    out_dir = args.out or (out_root / "eval")
    base = args.base_checkpoint or pretrained_checkpoint(out_root)
    volumes = [v.strip() for v in args.volumes.split(",") if v.strip()]
    summary = evaluate(
        EvalRequest(
            finetuned=args.checkpoint,
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
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
