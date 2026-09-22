"""Evaluate a fine-tuned SAM2 checkpoint on the val split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam2_segmentation_trainer.data.dataset import EMSegDataset, collate_em
from sam2_segmentation_trainer.data.manifest import index_volumes
from sam2_segmentation_trainer.data.splits import load_split
from sam2_segmentation_trainer.metrics import boundary_f1, iou_dice_pr, summarize_records
from sam2_segmentation_trainer.model import build_em_sam2, predict_masks
from sam2_segmentation_trainer.paths import data_root, output_root, pretrained_checkpoint


def _denorm_preview(image_chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = (image_chw.cpu() * std + mean).clamp(0, 1)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr


def _overlay_panel(gray_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = gray_rgb.copy()
    gt_panel = base.copy()
    pred_panel = base.copy()
    gt_m = gt.astype(bool)
    pred_m = pred.astype(bool)
    gt_panel[gt_m] = (0.6 * gt_panel[gt_m] + np.array([0, 180, 0]) * 0.4).astype(np.uint8)
    pred_panel[pred_m] = (0.6 * pred_panel[pred_m] + np.array([0, 180, 0]) * 0.4).astype(np.uint8)
    fp = pred_m & ~gt_m
    fn = gt_m & ~pred_m
    pred_panel[fp] = (0.4 * pred_panel[fp] + np.array([220, 30, 30]) * 0.6).astype(np.uint8)
    pred_panel[fn] = (0.4 * pred_panel[fn] + np.array([30, 80, 220]) * 0.6).astype(np.uint8)
    return Image.fromarray(np.concatenate([base, gt_panel, pred_panel], axis=1))


def evaluate(
    *,
    finetuned: Path,
    base_ckpt: Path,
    model_cfg: str,
    data_dir: Path,
    volumes: list[str],
    split_path: Path,
    image_size: int,
    threshold: float,
    out_dir: Path,
    grid_every: int,
    max_grids: int,
    num_workers: int,
) -> dict:
    examples = index_volumes(volumes=volumes, root=data_dir, skip_missing=True)
    split = load_split(split_path)
    _train_ex, val_ex = split.partition(examples)
    dataset = EMSegDataset(val_ex, split="val", image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_em,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_em_sam2(model_cfg, str(base_ckpt), device, train=False)
    state = torch.load(finetuned, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    records = []
    grid_dir = out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grids_written = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="eval")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            points = batch["point"].to(device)
            labels = batch["point_label"].to(device)
            pred, _iou = predict_masks(
                model, images, points, labels, image_size=image_size
            )
            pred_bin = (torch.sigmoid(pred) > threshold)[0].cpu().numpy().astype(np.uint8)
            gt = masks[0].cpu().numpy().astype(np.uint8)
            stats = iou_dice_pr(pred_bin, gt)
            stats["bf"] = boundary_f1(pred_bin, gt)
            meta = batch["meta"][0]
            stats.update(meta)
            records.append(stats)
            if grid_every > 0 and i % grid_every == 0 and grids_written < max_grids:
                preview = _denorm_preview(batch["image"][0])
                panel = _overlay_panel(preview, gt, pred_bin)
                name = f"{meta['volume']}_{meta['image_key']}_{meta['location_id']}.png"
                panel.save(grid_dir / name)
                grids_written += 1

    summary = summarize_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
