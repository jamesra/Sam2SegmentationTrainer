# SAM2 Fine-Tuning Plan for TEM/SEM Electron Microscopy

This document is a complete, self-contained brief for an AI coding assistant to implement fine-tuning of Meta's SAM2 (Segment Anything Model 2) on electron microscopy data. All code should be Python 3.10+. The implementer should work through sections in order.

## 1. Overview & Scope

This plan guides an AI implementer through every step needed to fine-tune SAM2 on a domain-specific corpus of Transmission Electron Microscopy (TEM) and Scanning Electron Microscopy (SEM) images. The user has **32,596 annotated mask examples** across multiple folders in SAM2 training format. This is a substantial dataset — roughly two orders of magnitude larger than most published medical/scientific SAM fine-tuning efforts — which changes several training decisions (see below). The deliverable is a fine-tuned model checkpoint (`.pt` or `.pth`) and a clean inference wrapper.

**What the implementer will produce:**

- A validated, augmented dataset ready for training
- A Python training script (`train_sam2_em.py`) using SAM2's official training framework
- A configuration YAML for hyperparameters
- Evaluation outputs (IoU, boundary F1, qualitative grid)
- An inference wrapper (`infer_sam2_em.py`) using the fine-tuned weights

**Key challenges of EM segmentation SAM2 must learn:**

- Images are typically 8-bit or 16-bit grayscale (SAM2 expects RGB — a conversion strategy is required)
- TEM images have low-contrast phase boundaries and overlapping projections
- SEM images have textured surfaces with variable depth-of-field artifacts
- Objects of interest range from sub-nanometre features (atomic columns) to micron-scale regions
- The training set may be small relative to natural image datasets SAM2 was trained on

## 2. Prerequisites & Environment Setup

**Hardware (minimum):** NVIDIA GPU with ≥16 GB VRAM (A100/H100 recommended for `sam2_hiera_large`; RTX 3090/4090 workable with `sam2_hiera_base_plus`). At least 32 GB system RAM. SSD storage.

**Python environment:**

```bash
conda create -n sam2_em python=3.11 -y
conda activate sam2_em

# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SAM2 (install from source to access training code)
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[dev]"

# Training utilities
pip install hydra-core omegaconf tensorboard tqdm albumentations scikit-image scipy
pip install opencv-python-headless Pillow
```

**Verify installation:**

```python
import torch, sam2
print(torch.__version__)          # should be 2.x
print(torch.cuda.is_available())  # must be True
print(sam2.__version__)
```

**Download pretrained weights** (choose one):

```bash
# Largest / best quality — fine-tune this unless VRAM is scarce
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Smaller alternative
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Store weights in `./checkpoints/` relative to the project root.

## 3. Data Validation & Preparation

**Expected SAM2 folder structure (validate before training):**

```
dataset_root/
├── JPEGImages/          # or PNGs — source images
├── Annotations/         # binary or indexed mask PNGs, same filename stem
└── ImageSets/
    └── Main/
        ├── train.txt
        └── val.txt
```

**Validation script** — write `validate_dataset.py`:

```python
from pathlib import Path
import numpy as np
from PIL import Image

def validate(root: str):
    root = Path(root)
    images = sorted((root / "JPEGImages").glob("*"))
    masks  = sorted((root / "Annotations").glob("*"))
    assert len(images) == len(masks), "Image/mask count mismatch"
    issues = []
    for img_p, msk_p in zip(images, masks):
        img = Image.open(img_p)
        msk = np.array(Image.open(msk_p))
        if img.size != Image.open(msk_p).size:
            issues.append(f"{img_p.name}: size mismatch")
        if msk.max() == 0:
            issues.append(f"{msk_p.name}: empty mask")
    return issues

if __name__ == "__main__":
    for folder in Path("data").iterdir():
        problems = validate(str(folder))
        if problems:
            for p in problems: print(p)
        else:
            print(f"{folder.name}: OK")
```

**Grayscale → RGB conversion (critical for EM data):** SAM2's image encoder was pretrained on RGB images. Convert grayscale EM images by replicating the single channel across all three: `img_rgb = np.stack([img_gray]*3, axis=-1)`. This should happen inside the dataset class, not as a pre-processing step on disk (keep originals).

**Train / validation split:** with 32,596 examples, use a 90/10 split (≈29,336 train / 3,260 val). A 90/10 ratio is appropriate because the validation set is already large enough for statistically stable metric estimates. Write split indices to `ImageSets/Main/train.txt` and `val.txt` (one filename stem per line, no extension). Stratify the split by dataset folder so each source is proportionally represented in both sets.

**Augmentation strategy for EM** — implement in `em_augment.py` using Albumentations:

```python
import albumentations as A

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # EM-specific: simulate scan noise
    A.GaussNoise(var_limit=(10, 50), p=0.4),
    # Simulate beam-induced contrast variation
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    # Simulate defocus/astigmatism blur
    A.Blur(blur_limit=(3, 7), p=0.3),
    A.ElasticTransform(alpha=30, sigma=5, p=0.2),  # sample distortion
    A.Resize(1024, 1024),  # SAM2 encoder input size
], additional_targets={"mask": "mask"})

val_transform = A.Compose([
    A.Resize(1024, 1024),
], additional_targets={"mask": "mask"})
```

**Do not apply** colour jitter or hue/saturation shifts — EM images have no chromatic content and these will mislead the model.

## 4. Model & Strategy Selection

**Recommended model:** `sam2.1_hiera_large` — best mask quality; use `sam2.1_hiera_base_plus` if VRAM < 16 GB.

**Fine-tuning strategy — parameter-efficient approach:**

SAM2 has three major components: the image encoder (Hiera backbone), the memory attention module, and the mask decoder. For a domain-shift fine-tune (natural → EM images), the recommended freeze strategy is:

| Component | Action | Rationale |
| --- | --- | --- |
| Hiera encoder — early stages (0–1) | Train | 32K examples warrant full adaptation; EM textures differ from natural images at all levels |
| Hiera encoder — late stages (2–3) | Train | High-level features need domain adaptation |
| Memory attention | Train from start | 32K examples provide sufficient signal; no staged unfreezing needed |
| Mask decoder | Always train | Direct output quality |
| Prompt encoder | Freeze | Keeps point/box prompts working correctly |

**Why full fine-tune?** With 32K+ examples, full fine-tuning is now viable and **recommended**. The dataset is large enough that catastrophic forgetting is unlikely. The selective freeze table above is the fallback for VRAM-constrained hardware; otherwise **unfreeze the entire model** (all encoder stages + memory attention + mask decoder) for maximum domain adaptation.

**Implementation — freeze logic in Python:**

```python
def configure_trainable_params(model, freeze_mode="none"):
    """
    freeze_mode:
      "none"    → train everything (recommended with 32K+ examples)
      "partial" → freeze encoder stages 0-1, train stages 2-3 + decoder
    """
    if freeze_mode == "none":
        # Full fine-tune — unfreeze everything except prompt encoder
        for p in model.parameters():
            p.requires_grad = True
        for p in model.sam_prompt_encoder.parameters():
            p.requires_grad = False
    elif freeze_mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.sam_mask_decoder.parameters():
            p.requires_grad = True
        for stage_idx in (2, 3):
            for p in model.image_encoder.trunk.blocks[stage_idx].parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params "
          f"({100*trainable/total:.0f}%) [{freeze_mode}]")
```

**If VRAM is limited (< 24 GB):** fall back to freezing stages 0–1 and training only stages 2–3 + decoder. With 32K examples the quality gap will be small. LoRA adapters are unnecessary at this data scale.

## 5. Training Configuration

Create `config/em_finetune.yaml` (loaded by Hydra):

```yaml
# config/em_finetune.yaml
training:
  output_dir: "./outputs/sam2_em_v1"
  num_epochs: 15               # 32K examples — fewer epochs needed than with small data
  batch_size: 4                 # per GPU; 8 if VRAM ≥40 GB
  num_workers: 8                # increase for 32K-image dataloader throughput
  seed: 42
  mixed_precision: true         # bfloat16 on Ampere+, float16 otherwise
  gradient_accumulation_steps: 4  # effective batch = 4 × 4 = 16
  gradient_clip: 1.0
  log_every_n_steps: 50
  val_every_n_epochs: 1
  save_every_n_epochs: 3
  early_stopping_patience: 5    # halt if val IoU stagnates for 5 epochs

optimizer:
  type: AdamW
  lr: 1.0e-4                    # can be more aggressive with 32K examples
  weight_decay: 0.01
  betas: [0.9, 0.999]

lr_schedule:
  type: cosine_with_warmup
  warmup_epochs: 1              # 1 epoch = ~7,334 steps at batch 4 — sufficient
  min_lr: 1.0e-6

differential_lr:
  encoder_lr_factor: 0.1        # encoder gets lr × 0.1

loss:
  focal_weight: 20.0
  dice_weight: 1.0
  iou_weight: 1.0

data:
  root: "./data"                # parent of all SAM2-format dataset folders
  image_size: 1024
  mask_threshold: 0.5
  train_split: 0.9              # 90/10 with 32K examples

model:
  checkpoint: "./checkpoints/sam2.1_hiera_large.pt"
  config: "sam2_hiera_l.yaml"
  freeze_mode: "none"           # "none" = full fine-tune (recommended w/ 32K)
                                # "partial" = freeze encoder stages 0-1 (VRAM fallback)
```

**Loss function rationale:** The combination of Focal loss (addresses class imbalance in small EM features) + Dice loss (penalises area mismatch) + IoU prediction loss (SAM2's auxiliary head) is consistent with SAM2's original training and well-validated for medical/scientific imaging segmentation tasks in the peer-reviewed literature (e.g., MedSAM, SAM-Med2D).

**Learning rate guidance with 32K examples:** 1e-4 base rate with 0.1× factor on the encoder. The larger dataset permits a more aggressive learning rate than the 5e-5 typical of small-data SAM fine-tuning. If training is unstable (loss spikes in the first 500 steps), halve the base rate. The gradient accumulation of 4 gives an effective batch size of 16, which stabilises gradients without requiring multi-GPU setups.

**Epoch count rationale:** At 29,336 training examples with effective batch 16, one epoch is ≈1,833 steps. With 15 epochs (≈27.5K total steps) and cosine decay, the model sees each example ≈15 times — sufficient for convergence without overfitting, given augmentation. Monitor val loss: if it plateaus by epoch 8–10, the early stopping will catch it.

**Learning rate guidance:** 5e-5 for the mask decoder; 5e-6 for encoder stages. If training is unstable (loss spikes), reduce to 1e-5 / 1e-6. With very small datasets, consider freezing the encoder entirely and using lr = 1e-4 on the decoder only.

## 6. Training Implementation

Write the following files. The implementer should place them in the project root.

**`dataset.py` — PyTorch Dataset for SAM2 format:**

```python
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from em_augment import train_transform, val_transform

class EMSegDataset(Dataset):
    def __init__(self, roots: list[str], split: str = "train"):
        """
        roots: list of dataset folder paths (each in SAM2/VOC format)
        split: 'train' or 'val'
        """
        self.samples = []
        for root in roots:
            root = Path(root)
            split_file = root / "ImageSets" / "Main" / f"{split}.txt"
            stems = split_file.read_text().strip().splitlines()
            for stem in stems:
                img_candidates = list((root / "JPEGImages").glob(f"{stem}.*"))
                msk_candidates = list((root / "Annotations").glob(f"{stem}.*"))
                if img_candidates and msk_candidates:
                    self.samples.append((str(img_candidates[0]), str(msk_candidates[0])))
        self.transform = train_transform if split == "train" else val_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        # Load and convert to 3-channel (EM images are typically grayscale)
        img = np.array(Image.open(img_path).convert("L"))
        img = np.stack([img, img, img], axis=-1)  # H x W x 3
        msk = np.array(Image.open(msk_path).convert("L"))  # H x W
        msk = (msk > 127).astype(np.uint8)  # binarise
        augmented = self.transform(image=img, mask=msk)
        img = augmented["image"].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # 3 x H x W
        msk = torch.from_numpy(augmented["mask"]).long()
        return img, msk
```

**`train_sam2_em.py` — main training script:**

```python
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from dataset import EMSegDataset

# ---- helpers ----------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred   = torch.sigmoid(pred).flatten(1)
    target = target.float().flatten(1)
    inter  = (pred * target).sum(1)
    return 1 - (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)

def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def configure_trainable_params(model: SAM2Base, unfreeze_stages=(2, 3)):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    for stage_idx in unfreeze_stages:
        stage = model.image_encoder.trunk.blocks[stage_idx]
        for p in stage.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

# ---- main -------------------------------------------------------------------

def main(cfg):
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_sam2(cfg.model.config, cfg.model.checkpoint, device=device)
    configure_trainable_params(model, cfg.model.unfreeze_encoder_stages)

    # Datasets
    data_roots = [str(p) for p in Path(cfg.data.root).iterdir() if p.is_dir()]
    train_ds = EMSegDataset(data_roots, split="train")
    val_ds   = EMSegDataset(data_roots, split="val")
    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                          shuffle=True, num_workers=cfg.training.num_workers,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=1,
                          shuffle=False, num_workers=cfg.training.num_workers)

    # Differential learning rates
    decoder_params = list(model.sam_mask_decoder.parameters())
    encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": decoder_params, "lr": cfg.optimizer.lr},
        {"params": encoder_params, "lr": cfg.optimizer.lr * cfg.differential_lr.encoder_lr_factor},
    ], weight_decay=cfg.optimizer.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.training.num_epochs)
    scaler    = GradScaler(enabled=cfg.training.mixed_precision)

    best_val_iou = 0.0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0.0
        for step, (imgs, masks) in enumerate(train_dl):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.training.mixed_precision):
                # SAM2 image encoding
                backbone_out = model.forward_image(imgs)
                _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
                feats = [f[-1].unsqueeze(0) for f in vision_feats]

                # Use centre-of-mass point prompts derived from GT mask
                # (approximates oracle prompting; replace with your prompt strategy)
                b, h, w = masks.shape
                prompt_points, prompt_labels = [], []
                for bi in range(b):
                    ys, xs = torch.where(masks[bi] > 0)
                    if len(ys):
                        cy = ys.float().mean().item()
                        cx = xs.float().mean().item()
                        prompt_points.append([[cx / w, cy / h]])
                        prompt_labels.append([[1]])
                    else:
                        prompt_points.append([[0.5, 0.5]])
                        prompt_labels.append([[0]])
                pts = torch.tensor(prompt_points, device=device)   # B x 1 x 2
                lbs = torch.tensor(prompt_labels, device=device)   # B x 1

                sparse_embed, dense_embed = model.sam_prompt_encoder(
                    points=(pts, lbs), boxes=None, masks=None)
                low_res_masks, iou_preds, _, _ = model.sam_mask_decoder(
                    image_embeddings=feats[0],
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embed,
                    dense_prompt_embeddings=dense_embed,
                    multimask_output=False,
                )
                # Upsample to input size
                pred_masks = F.interpolate(low_res_masks, size=(h, w),
                                           mode="bilinear", align_corners=False)
                pred_masks = pred_masks[:, 0]  # B x H x W

                loss_focal = focal_loss(pred_masks, masks)
                loss_dice  = dice_loss(pred_masks, masks).mean()
                # IoU head loss
                with torch.no_grad():
                    gt_iou = ((torch.sigmoid(pred_masks) > 0.5) & masks.bool()).sum((-2,-1)).float() / \
                             (((torch.sigmoid(pred_masks) > 0.5) | masks.bool()).sum((-2,-1)).float() + 1e-6)
                loss_iou = F.mse_loss(iou_preds[:, 0], gt_iou)

                loss = (cfg.loss.focal_weight * loss_focal
                      + cfg.loss.dice_weight  * loss_dice
                      + cfg.loss.iou_weight   * loss_iou)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if step % cfg.training.log_every_n_steps == 0:
                print(f"Epoch {epoch} | step {step} | loss {loss.item():.4f}")

        scheduler.step()

        # Validation
        if epoch % cfg.training.val_every_n_epochs == 0:
            model.eval()
            ious = []
            with torch.no_grad():
                for imgs, masks in val_dl:
                    imgs, masks = imgs.to(device), masks.to(device)
                    backbone_out = model.forward_image(imgs)
                    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
                    feats = [f[-1].unsqueeze(0) for f in vision_feats]
                    b, h, w = masks.shape
                    pts = torch.tensor([[[0.5, 0.5]]], device=device)
                    lbs = torch.tensor([[[1]]], device=device)
                    sparse_embed, dense_embed = model.sam_prompt_encoder(
                        points=(pts, lbs), boxes=None, masks=None)
                    low_res_masks, _, _, _ = model.sam_mask_decoder(
                        image_embeddings=feats[0],
                        image_pe=model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embed,
                        dense_prompt_embeddings=dense_embed,
                        multimask_output=False,
                    )
                    pred = F.interpolate(low_res_masks, size=(h, w),
                                         mode="bilinear", align_corners=False)
                    pred_bin = (torch.sigmoid(pred[:, 0]) > 0.5).long()
                    inter = (pred_bin & masks.bool()).sum().item()
                    union = (pred_bin | masks.bool()).sum().item()
                    ious.append(inter / (union + 1e-6))
            val_iou = np.mean(ious)
            print(f"✔ Epoch {epoch} | val IoU {val_iou:.4f}")
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(),
                           output_dir / "best_model.pt")
                print(f"  Saved best model (IoU {best_val_iou:.4f})")

        if epoch % cfg.training.save_every_n_epochs == 0:
            torch.save(model.state_dict(),
                       output_dir / f"checkpoint_epoch{epoch}.pt")

    print(f"Training complete. Best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="em_finetune")
    def run(cfg: DictConfig):
        main(cfg)

    run()
```

**Run training:**

```bash
python train_sam2_em.py
# Override any YAML param on the command line:
python train_sam2_em.py training.num_epochs=50 training.batch_size=4
```

**Monitor with TensorBoard** (add `SummaryWriter` calls at the marked log points, writing `train/loss`, `val/iou`):

```bash
tensorboard --logdir outputs/
```

## 7. Evaluation & Validation

**Quantitative metrics — write `evaluate.py`:**

| Metric | Why it matters for EM |
| --- | --- |
| IoU (Jaccard) | Standard segmentation accuracy; report mean over classes |
| Dice coefficient | More sensitive to small-object errors; critical for sub-nm features |
| Boundary F1 (BF) | Penalises rough mask edges; important for membrane/grain boundary tasks |
| Precision / Recall | Reveals whether the model over- or under-segments |

Compute BF using `skimage.segmentation.find_boundaries` with a 2-px tolerance ring. Report all metrics per dataset folder and in aggregate.

**Evaluation script skeleton:**

```python
import numpy as np
from skimage.segmentation import find_boundaries

def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2) -> float:
    pred_b = find_boundaries(pred_bin, mode="outer")
    gt_b   = find_boundaries(gt_bin,   mode="outer")
    # Dilate for tolerance
    from scipy.ndimage import binary_dilation
    struct = np.ones((2*tolerance+1, 2*tolerance+1))
    pred_b_d = binary_dilation(pred_b, structure=struct)
    gt_b_d   = binary_dilation(gt_b,   structure=struct)
    prec = (pred_b & gt_b_d).sum() / (pred_b.sum() + 1e-6)
    rec  = (gt_b   & pred_b_d).sum() / (gt_b.sum()  + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)

def evaluate_model(model, val_dl, device):
    metrics = {"iou": [], "dice": [], "bf": []}
    model.eval()
    # ... (run inference loop, collect pred and gt, compute metrics)
    return {k: np.mean(v) for k, v in metrics.items()}
```

**Qualitative checks — generate a visual grid:**

For every 10th validation image, save a 3-panel PNG: original EM image | ground truth mask overlay | predicted mask overlay. Colour the overlay at 40% opacity. Use a distinct colour for false positives (red) and false negatives (blue) to diagnose failure modes.

**Common EM failure modes to check:**

- Over-segmentation of surface contamination in SEM images
- Under-segmentation at low-contrast grain boundaries in TEM
- Holes inside large objects (common with Focal loss — add a connectivity post-processing step)
- Prompt sensitivity: test with 1, 3, and 5 point prompts to measure robustness

**Stopping criterion:** halt training if val IoU does not improve for 5 consecutive validation epochs (early stopping). With 32K examples and 15 max epochs, expect convergence around epoch 8–12. The best checkpoint is the one to ship.

**Statistical confidence with 3,260 val examples:** report 95% bootstrap confidence intervals for IoU and Dice (1,000 resamples) to quantify how tight the metric estimates are. With this validation set size, expect CI widths of ±0.5–1%.

## 8. Export & Deployment

**Save the fine-tuned checkpoint:**

The training script saves `best_model.pt` — this contains the full model `state_dict`. To reload:

```python
from sam2.build_sam import build_sam2
model = build_sam2("sam2_hiera_l.yaml",
                   "./checkpoints/sam2.1_hiera_large.pt",
                   device="cuda")
model.load_state_dict(torch.load("outputs/sam2_em_v1/best_model.pt"))
model.eval()
```

**Inference wrapper — write `infer_sam2_em.py`:**

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class EMPredictor:
    """Drop-in wrapper around SAM2ImagePredictor using the fine-tuned weights."""

    def __init__(self,
                 checkpoint: str = "outputs/sam2_em_v1/best_model.pt",
                 model_cfg:  str = "sam2_hiera_l.yaml",
                 base_ckpt:  str = "checkpoints/sam2.1_hiera_large.pt",
                 device:     str = "cuda"):
        model = build_sam2(model_cfg, base_ckpt, device=device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.predictor = SAM2ImagePredictor(model)
        self.device    = device

    def set_image(self, image_path: str):
        """Load and set an EM image. Handles grayscale automatically."""
        img = Image.open(image_path).convert("L")
        arr = np.array(img)
        rgb = np.stack([arr, arr, arr], axis=-1)  # H x W x 3
        self.predictor.set_image(rgb)

    def predict(self,
                point_coords: np.ndarray | None = None,
                point_labels: np.ndarray | None = None,
                box:          np.ndarray | None = None,
                multimask:    bool = True):
        """
        point_coords: Nx2 array of (x, y) coordinates in pixel space
        point_labels: N array of 1 (foreground) or 0 (background)
        box: [x0, y0, x1, y1] bounding box
        Returns: masks (NxHxW), scores, logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask,
        )
        return masks, scores, logits


# Example usage
if __name__ == "__main__":
    predictor = EMPredictor()
    predictor.set_image("example_tem.tif")
    # Click a point inside the feature of interest
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[512, 400]]),
        point_labels=np.array([1]),
    )
    best = masks[scores.argmax()]
    Image.fromarray(best.astype(np.uint8) * 255).save("output_mask.png")
    print("Saved output_mask.png")
```

**Optional: ONNX export** (for integration with C#/MATLAB pipelines):

```bash
pip install onnx onnxruntime-gpu
```

```python
# Export the image encoder
torch.onnx.export(
    model.image_encoder,
    torch.randn(1, 3, 1024, 1024, device="cuda"),
    "sam2_em_encoder.onnx",
    input_names=["image"],
    output_names=["features"],
    dynamic_axes={"image": {0: "batch"}},
    opset_version=17,
)
print("Encoder exported to sam2_em_encoder.onnx")
```

Note: The mask decoder is not straightforward to export due to dynamic prompt encoding; keep it in PyTorch or use TorchScript (`torch.jit.script`) for the full pipeline in a C# Torch.NET deployment.

## 9. EM-Specific Prompt Engineering

SAM2's interactive segmentation quality depends heavily on how prompts are chosen. EM images have domain-specific characteristics that demand a deliberate prompting strategy.

**Point prompts (most common in practice):**

- **For TEM** (bright-field: features are darker regions): click inside the darkest core of the feature. Avoid the diffuse halo around particles.
- **For SEM** (features are often bright against a dark substrate): click the brightest point of the surface feature. Add a background point (label 0) at the surrounding substrate to suppress over-expansion.
- Use 3 foreground points in a triangle inside large objects rather than 1 centre point — this substantially improves IoU for irregular EM features (consistent with the multi-point prompting findings in Kirillov et al. 2023 and MedSAM evaluations).

**Box prompts (best for known ROI size):**

Wrap a bounding box tightly around the feature with \~5% padding. Box prompts outperform single-point prompts for compact objects (nanoparticles, pores, precipitates). In automated pipelines, generate boxes from a lightweight detector (e.g. YOLO-v8 fine-tuned on EM data) and feed them to SAM2.

```python
# Example: box prompt for a nanoparticle at pixel coords
box = np.array([x0 - 5, y0 - 5, x1 + 5, y1 + 5])  # small padding
masks, scores, _ = predictor.predict(box=box, multimask_output=True)
best_mask = masks[scores.argmax()]
```

**Multi-mask output:** always use `multimask_output=True` and select the mask with the highest IoU score. SAM2 returns 3 candidates (whole-object, part, sub-part) — for EM segmentation the highest-score mask is almost always the correct grain/particle/region.

**Mask refinement prompt (iterative):** after an initial mask, feed it back as a mask prompt and add corrective click points for missed or over-included regions:

```python
# Iterative refinement loop
masks, scores, logits = predictor.predict(
    point_coords=np.array([[cx, cy]]), point_labels=np.array([1]),
    multimask_output=True)
for missed_x, missed_y in user_corrections_fg:
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[cx, cy], [missed_x, missed_y]]),
        point_labels=np.array([1, 1]),
        mask_input=logits[scores.argmax()][None],  # feed previous logit
        multimask_output=False)
```

**Automated batch inference (no human prompts):** for fully automated pipelines, generate automatic masks with `SAM2AutomaticMaskGenerator`. Tune `pred_iou_thresh=0.88`, `stability_score_thresh=0.92`, and `min_mask_region_area` to the expected feature size in pixels. Post-filter by area and circularity for nanoparticle counting applications.

**Normalisation before inference:** EM images often have 12-bit or 16-bit dynamic range. Normalise to 8-bit before setting the image:

```python
def normalise_em(arr: np.ndarray) -> np.ndarray:
    """Percentile-based normalisation robust to hot/dead pixels."""
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
```
