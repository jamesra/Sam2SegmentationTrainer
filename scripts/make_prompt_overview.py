"""Draw one three-panel overview image per prompt family (docs/prompt_overview.md).

Each image shows, side by side, for one real crop of a training volume:

1. INPUT: the crop with the prompt drawn on it (points, box, seed or mask to correct, negative neighbours);
2. INPUT + PREDICTION: the same, with the outline of the model's predicted mask;
3. GROUND TRUTH: the crop with the true mask.

Nothing about the prompts is re-implemented here. The prompt of every family is built by the trainer's own
dataset code (``EMSegDataset._mixed_item`` with the family forced, which draws the annotator-like and pull-through
discs as training does and calls ``build_prompt``); the prediction is ``validation.predict_batch`` on the same
batch. The adjacent-section shape comes from the trainer's own loader too: ``build_tile_context`` reads
``<VOL>/shapes.jsonl`` and, given ``--registration-root`` and ``--corrections-root`` (the defaults are the trainer
container's ``/storage4`` and ``/corrections``), carries each linked neighbour outline through the volume's
registration as training does (``prompt.adjacent_poly.placement: projected``); ``TileContext.adjacent_draw`` erodes it
and applies the overlap rule, and ``build_prompt`` uses it where it lands. Without that data the volume falls back
to the centroid placement (``fit_adjacent_poly_seed`` scales the shape inside the cell); the image title says which.

To add a family, add one ``Family`` entry to ``FAMILIES``.

Run inside the trainer container, from the repo root (/workspace)::

    python scripts/make_prompt_overview.py --out docs/img/prompts --volume RC2

Without ``--checkpoint`` (or with ``--no-predict``) the prediction panel says so and the other two are still drawn.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CHECKPOINT = "/storage4/Sam2Trainer/runs/sam2_em_v4_centred/best_model.pt"
DEFAULT_LABEL = "illustrative: v4 epoch-0 weights, for format only, not a quality claim"

PANEL_W = 590
GAP = 6
IMAGE_SIZE = 1024

GREEN = (0, 200, 60)
ORANGE = (255, 150, 0)
RED = (225, 40, 40)
CYAN = (0, 200, 255)
MAGENTA = (255, 0, 220)
YELLOW = (255, 230, 0)


@dataclass(frozen=True)
class Family:
    """One prompt family of the overview.

    ``prompt_type`` is the trainer's ``PromptType`` name forced for the sample; ``validation_family`` instead names
    a family of ``VAL_PROMPTS`` that only the validation path builds. ``source`` picks the dataset: ``plain``
    (the training dataset, adjacent shapes from shapes.jsonl included) or ``families`` (validation families). ``prob_key`` is the
    ``prompt.*`` key of the training probability in em_finetune.yaml (None: not drawn in training).
    """

    name: str
    prompt_type: str | None = None
    validation_family: str | None = None
    source: str = "plain"
    prob_key: str | None = None
    note: str = ""
    """Extra text for the image title."""


FAMILIES = [
    Family("points", "POINTS", prob_key="prob_points_only"),
    Family("box", "BOX", prob_key="prob_box_only"),
    Family("box_points", "BOX_POINTS", prob_key="prob_box_points"),
    Family("circle_seed", "CIRCLE_SEED", prob_key="prob_circle_seed", note="inscribed circle x0.95 shown"),
    Family("circle_annotator", "CIRCLE_ANNOTATOR", prob_key="prob_circle_annotator", note="annotator-like circle"),
    Family(
        "circle_pull_through", "CIRCLE_PULL_THROUGH", prob_key="prob_circle_pull_through",
        note="circle across the cell edge",
    ),
    Family(
        "adjacent_poly_seed", "ADJACENT_POLY_SEED", prob_key="prob_adjacent_poly_seed",
        note="neighbour polygon from shapes.jsonl",
    ),
    Family("subset_seed", "SUBSET_SEED", prob_key="prob_subset_seed"),
    Family("correction", "CORRECTION", prob_key="prob_correction"),
    Family("near_correct", "NEAR_CORRECT", prob_key="prob_near_correct"),
]


# ---------------------------------------------------------------------------------------------------------------
# Drawing


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    return mask & ~binary_erosion(mask, iterations=width)


def blend(rgb: np.ndarray, where: np.ndarray, colour, alpha) -> None:
    """Alpha-blend ``colour`` into ``rgb`` (in place) where ``where``; ``alpha`` is a scalar or a per-pixel array."""
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 0:
        a = np.full(where.shape, float(a), dtype=np.float32)
    a = np.where(where, a, 0.0)[..., None]
    rgb[:] = (rgb * (1.0 - a) + np.array(colour, dtype=np.float32) * a).astype(np.uint8)


def crop_window(extents: np.ndarray, size: int = IMAGE_SIZE, minimum: int = 320) -> tuple[int, int, int]:
    """Square window ``(x0, y0, side)`` around the ``(x1, y1, x2, y2)`` rows of ``extents``, kept inside the tile."""
    x1, y1 = extents[:, 0].min(), extents[:, 1].min()
    x2, y2 = extents[:, 2].max(), extents[:, 3].max()
    side = max(x2 - x1, y2 - y1)
    side = int(min(size, max(minimum, side * 1.18 + 40)))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = int(np.clip(round(cx - side / 2), 0, size - side))
    y0 = int(np.clip(round(cy - side / 2), 0, size - side))
    return x0, y0, side


def mask_extent(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def prompt_extent(entry: dict) -> list[list[float]]:
    rows = [mask_extent(entry["target"])]
    if entry["input_mask"] is not None and entry["input_mask"].any():
        rows.append(mask_extent(entry["input_mask"]))
    if entry["box"] is not None:
        rows.append([float(v) for v in entry["box"]])
    for x, y in entry["points"]:
        rows.append([float(x) - 6, float(y) - 6, float(x) + 6, float(y) + 6])
    return rows


def draw_prompt(entry: dict, window: tuple[int, int, int], dense_full: np.ndarray, scale: float) -> Image.Image:
    """The crop with the prompt drawn on it, ``PANEL_W`` pixels square."""
    x0, y0, side = window
    rgb = entry["rgb"][y0 : y0 + side, x0 : x0 + side]
    panel = np.array(Image.fromarray(rgb).resize((PANEL_W, PANEL_W), Image.LANCZOS))
    dense = np.array(
        Image.fromarray(dense_full[y0 : y0 + side, x0 : x0 + side]).resize((PANEL_W, PANEL_W), Image.BILINEAR)
    )
    if entry["has_dense"]:
        high = float(dense_full.max())
        low = float(-dense_full.min())
        if high > 0:
            colour = ORANGE if entry["kind"] == "correction" else GREEN
            blend(panel, dense > 0.05, colour, 0.55 * np.clip(dense / high, 0, 1))
        if low > 0:
            blend(panel, dense < -0.05, RED, 0.5 * np.clip(-dense / low, 0, 1))
        mask = entry["input_mask"]
        if mask is not None and mask.any():
            crop = mask[y0 : y0 + side, x0 : x0 + side].astype(np.uint8) * 255
            resized = np.array(Image.fromarray(crop).resize((PANEL_W, PANEL_W), Image.NEAREST)) > 0
            blend(panel, outline(resized, 2), ORANGE if entry["kind"] == "correction" else GREEN, 1.0)
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    if entry["box"] is not None:
        bx1, by1, bx2, by2 = ((float(v) - o) * scale for v, o in zip(entry["box"], (x0, y0, x0, y0)))
        draw.rectangle([bx1, by1, bx2, by2], outline=CYAN, width=3)
    for (px, py), label in zip(entry["points"], entry["labels"]):
        cx, cy = (px - x0) * scale, (py - y0) * scale
        if not (-10 <= cx <= PANEL_W + 10 and -10 <= cy <= PANEL_W + 10):
            continue
        colour = GREEN if label == 1 else RED
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=colour, outline=(255, 255, 255), width=2)
    return image


def draw_outline_on(image: Image.Image, mask: np.ndarray, window: tuple[int, int, int], colour, width: int = 3):
    x0, y0, side = window
    crop = mask[y0 : y0 + side, x0 : x0 + side].astype(np.uint8) * 255
    resized = np.array(Image.fromarray(crop).resize((PANEL_W, PANEL_W), Image.NEAREST)) > 0
    arr = np.array(image)
    blend(arr, outline(resized, width), colour, 1.0)
    return Image.fromarray(arr)


def draw_truth(entry: dict, window: tuple[int, int, int]) -> Image.Image:
    x0, y0, side = window
    rgb = entry["rgb"][y0 : y0 + side, x0 : x0 + side]
    panel = np.array(Image.fromarray(rgb).resize((PANEL_W, PANEL_W), Image.LANCZOS))
    crop = entry["target"][y0 : y0 + side, x0 : x0 + side].astype(np.uint8) * 255
    mask = np.array(Image.fromarray(crop).resize((PANEL_W, PANEL_W), Image.NEAREST)) > 0
    blend(panel, mask, YELLOW, 0.30)
    blend(panel, outline(mask, 3), YELLOW, 1.0)
    return Image.fromarray(panel)


def text_panel(message: str) -> Image.Image:
    image = Image.new("RGB", (PANEL_W, PANEL_W), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    f = font(18)
    y = PANEL_W // 2 - 30
    for line in re.findall(r".{1,44}(?:\s|$)", message):
        draw.text((24, y), line.strip(), fill=(235, 235, 235), font=f)
        y += 26
    return image


LEGEND = [
    (GREEN, "seed or positive point"),
    (ORANGE, "mask to correct"),
    (RED, "negative neighbour or point"),
    (CYAN, "box"),
    (MAGENTA, "predicted outline"),
    (YELLOW, "ground truth"),
]


def compose(title: str, subtitle: str, panels: list[Image.Image], labels: list[tuple[str, str]]) -> Image.Image:
    head, bar, legend_h = 58, 48, 30
    width = 3 * PANEL_W + 4 * GAP
    height = head + bar + PANEL_W + legend_h + GAP
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((GAP + 4, 6), title, fill=(255, 255, 255), font=font(22))
    draw.text((GAP + 4, 34), subtitle, fill=(190, 190, 190), font=font(15))
    for i, (panel, (first, second)) in enumerate(zip(panels, labels)):
        x = GAP + i * (PANEL_W + GAP)
        draw.text((x + 4, head + 2), first, fill=(255, 255, 255), font=font(18))
        if second:
            draw.text((x + 4, head + 26), second, fill=(255, 190, 90), font=font(13))
        canvas.paste(panel, (x, head + bar))
    x = GAP + 4
    y = head + bar + PANEL_W + 8
    small = font(14)
    for colour, text in LEGEND:
        draw.rectangle([x, y + 2, x + 14, y + 16], fill=colour)
        draw.text((x + 20, y), text, fill=(210, 210, 210), font=small)
        x += 20 + int(draw.textlength(text, font=small)) + 22
    return canvas


# ---------------------------------------------------------------------------------------------------------------
# Building the prompts with the trainer's own code


def adjacent_note(tiles, volume: str, overlap: float | None) -> str:
    """Title note for adjacent_poly_seed: which placement drew it (the production path when projected)."""
    record = tiles.projection.get(volume) or {}
    if record.get("placement") == "projected":
        residual = "stos + residual" if record.get("convention") not in (None, "none") else "stos only"
        share = "" if overlap is None else f", {overlap:.0%} on the cell"
        return f"neighbour polygon projected ({residual}), eroded{share}"
    return "neighbour shape fitted inside (centroid fallback: no registration data)"


def build_entry(family: Family, datasets: dict, example, index: int) -> dict:
    """The trainer's item for ``family`` on ``example``, reduced to what is drawn. Raises KeyError when the family
    cannot be drawn for this sample (it fell back to another family, or the sample is unusable)."""
    from sam2_segmentation_trainer.data.prompts import PromptType
    from sam2_segmentation_trainer.overlays import denorm_preview

    dataset = datasets[family.source]
    if family.validation_family:
        item = dataset._families_item(example)
        wanted = PromptType.CIRCLE_SEED.value
    else:
        forced = PromptType[family.prompt_type]
        item = dataset._mixed_item(example, index, forced=forced)
        wanted = forced.value
    variant = item["variants"][0]
    if variant["prompt_type"] != wanted:
        raise KeyError(f"{family.name} fell back to {variant['prompt_type']}")
    dense = variant["dense_mask"]
    box = variant["boxes"]
    input_mask = variant["prompt_input"]
    kind = family.name
    return {
        "family": family,
        "item": item,
        "kind": "correction" if family.name in ("correction", "near_correct") else kind,
        "rgb": denorm_preview(item["image"]),
        "target": item["mask"].numpy() > 0,
        "points": variant["point"].numpy().reshape(-1, 2),
        "labels": variant["point_label"].numpy().reshape(-1),
        "box": None if box is None else box.numpy(),
        "has_dense": dense is not None,
        "dense": None if dense is None else dense.numpy().astype(np.float32),
        "input_mask": None if input_mask is None else input_mask.numpy() > 0,
        "adjacent_overlap": variant.get("adjacent_overlap"),
    }


def pick_sample(args, examples, val_examples, tiles, datasets):
    """The first sample (in a fixed shuffled order) on which every family can be drawn, and its entries."""
    rng = np.random.default_rng(args.seed)
    pool = list(val_examples if val_examples else examples)
    if args.sample:
        key = args.sample
        pool = [e for e in pool if f"{e.image_key}:{e.location_id}" == key]
        if not pool:
            raise SystemExit(f"--sample {key} is not in the {args.volume} examples")
    else:
        rng.shuffle(pool)
    tried = 0
    for example in pool:
        if example.width not in (None, IMAGE_SIZE) or example.height not in (None, IMAGE_SIZE):
            continue
        frac = example.area_frac
        if not args.sample and (frac is None or not 0.04 <= frac <= 0.25):
            continue
        key = (example.volume, example.image_key)
        if not args.sample:
            if len(tiles.neighbor_ids(key)) < 2 or tiles.circle_for(key, example.location_id) is None:
                continue
        if not args.sample and tiles.adjacent_shape(key, example.location_id, shape=(IMAGE_SIZE, IMAGE_SIZE)) is None:
            continue  # no linked neighbour outline: adjacent_poly_seed cannot be drawn on this sample
        tried += 1
        try:
            entries = [build_entry(f, datasets, example, 0) for f in FAMILIES]
        except KeyError as exc:
            print(f"skip {example.image_key} {example.location_id}: {exc}")
            continue
        if not args.sample and not any(
            (e["labels"] == 0).any() or (e["has_dense"] and e["dense"].min() < 0) for e in entries
        ):
            continue  # want a negative neighbour visible somewhere
        return example, entries
    raise SystemExit("no sample could be drawn for every family; pass --sample IMAGEKEY:LOCATION_ID")


def predict_all(entries, args):
    """Predicted binary masks (or None and a reason) for each entry, from the real model and ``predict_batch``."""
    import torch

    from sam2_segmentation_trainer.data.dataset import collate_em
    from sam2_segmentation_trainer.export_serve import load_finetuned_model
    from sam2_segmentation_trainer.paths import pretrained_checkpoint
    from sam2_segmentation_trainer.validation import predict_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned_model(Path(args.checkpoint), pretrained_checkpoint(), args.model_config, str(device))
    predictions = []
    with torch.no_grad():
        for entry in entries:
            batch = collate_em([entry["item"]])
            logits = predict_batch(model, batch, device, IMAGE_SIZE)
            predictions.append((torch.sigmoid(logits)[0] > args.threshold).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-panel overview image per prompt family")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "img" / "prompts")
    parser.add_argument("--volume", default="RC2", help="RC1 and RPC2 have almost no linked shapes, so adjacent_poly_seed is drawn from RC2 by default")
    parser.add_argument("--data-root", type=Path, default=Path("/data-local/v3"))
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_CHECKPOINT).parent, help="for split.json and the prompt encoding record")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-predict", action="store_true", help="draw input and ground truth only")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="caption of the prediction panel")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0, help="sample selection and dataset seed")
    parser.add_argument("--sample", default=None, help="IMAGEKEY:LOCATION_ID instead of the automatic choice")
    parser.add_argument("--families", default=None, help="comma-separated subset of family names")
    parser.add_argument("--registration-root", default="/storage4",
                        help="data.registration_root for the projected adjacent_poly_seed ('' for none)")
    parser.add_argument("--corrections-root", default="/corrections",
                        help="data.corrections_root (the published residual fields; '' for none)")
    args = parser.parse_args(argv)
    started = time.time()

    from sam2_segmentation_trainer import train
    from sam2_segmentation_trainer.data.dataset import EMSegDataset
    from sam2_segmentation_trainer.data.manifest import index_volumes, take_model_sized
    from sam2_segmentation_trainer.data.prompts import PromptConfig
    from sam2_segmentation_trainer.data.splits import load_split
    from sam2_segmentation_trainer.data.tiles import build_tile_context
    from sam2_segmentation_trainer.encoding_record import read_record, resolve_encoding

    record, source = read_record(args.run_dir)
    resolved = resolve_encoding(record, source=source)
    cfg = train.load_train_cfg(resolved.cfg_overrides())
    prompt_cfg = PromptConfig.from_cfg(cfg)

    examples = index_volumes([args.volume], args.data_root, skip_missing=True)
    examples, _wrong = take_model_sized(examples, IMAGE_SIZE)
    split_path = args.run_dir / "split.json"
    val_examples = load_split(split_path).partition(examples)[1] if split_path.is_file() else []
    corrections = args.corrections_root if args.corrections_root and Path(args.corrections_root).is_dir() else None
    if args.corrections_root and corrections is None:
        print(f"no {args.corrections_root}: adjacent outlines are projected through the stos only, without the residual")
    tiles = build_tile_context(
        examples, circle_scale=resolved.circle_diameter_scale,
        adjacent_placement=prompt_cfg.adjacent_poly.placement,
        registration_root=args.registration_root or None, corrections_root=corrections,
        adjacent_residual=str(cfg.data.get("adjacent_residual", "auto")),
    )

    def dataset(tile_context, **extra):
        return EMSegDataset(
            examples, split="val", image_size=IMAGE_SIZE, tiles=tile_context, prompt_cfg=prompt_cfg,
            seed=args.seed, circle_scale=resolved.circle_diameter_scale, **extra,
        )

    datasets = {
        "plain": dataset(tiles),
        "families": dataset(tiles, val_families=["circle_annotator"]),
    }
    example, entries = pick_sample(args, examples, val_examples, tiles, datasets)
    print(f"sample {example.image_key} location {example.location_id} (z={example.z})")

    # One window for every family, so the figures are comparable.
    rows = [row for entry in entries for row in prompt_extent(entry)]
    window = crop_window(np.array(rows))
    scale = PANEL_W / window[2]
    if args.families:  # sample and window come from every family, so a subset still draws the same crop
        wanted = [n.strip() for n in args.families.split(",")]
        entries = [e for e in entries if e["family"].name in wanted]

    predictions: list = [None] * len(entries)
    reason = ""
    if args.no_predict:
        reason = "not run (--no-predict)"
    else:
        try:
            predictions = predict_all(entries, args)
        except Exception as exc:  # report, never fabricate
            reason = f"not run: {type(exc).__name__}: {str(exc)[:100]}"
            print(f"prediction failed: {exc!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    for entry, pred in zip(entries, predictions):
        family = entry["family"]
        probability = "validation only" if family.prob_key is None else f"training probability {float(cfg.prompt[family.prob_key]):.0%}"
        dense_full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        if entry["has_dense"]:
            import cv2

            dense_full = cv2.resize(entry["dense"], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        first = draw_prompt(entry, window, dense_full, scale)
        if pred is None:
            second = text_panel(reason or "prediction missing")
        else:
            second = draw_outline_on(first, pred, window, MAGENTA)
        third = draw_truth(entry, window)
        note = adjacent_note(tiles, args.volume, entry["adjacent_overlap"]) if family.name == "adjacent_poly_seed" else family.note
        title = f"{family.name}  -  {probability}" + (f"  -  {note}" if note else "")
        subtitle = (
            f"{args.volume} {example.image_key}  location {example.location_id}  "
            f"crop x{window[0]} y{window[1]} {window[2]} px (shown at {PANEL_W} px)"
        )
        image = compose(
            title, subtitle, [first, second, third],
            [("1  INPUT", ""), ("2  INPUT + PREDICTION", args.label), ("3  GROUND TRUTH", "")],
        )
        path = args.out / f"{family.name}.png"
        image.save(path, optimize=True)
        print(f"wrote {path} ({image.width}x{image.height})")
    print(f"done in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
