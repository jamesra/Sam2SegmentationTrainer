# Sam2Trainer status

The current state, dated. Rules and specs say how things work; this file says where things stand. Check every entry
against the logs before acting on it (the `sam2trainer-startup` skill), and update it when it has changed.

## Runs (checked 2026-11-16)

- **Model v5** (`sam2_em_v5`, trained on training set v3, validated on validation set v5).
  - Finished 2026-10-31: early stopping at epoch 10.
  - Best epoch 7 (model v5.7): core recall 0.9732, IoU 0.8943, spill 0.026, merge rate 0.000, fragmentation 0.042.
  - Run directory: `/storage4/Sam2Trainer/runs/sam2_em_v5`; log at `/outputs/sam2_em_v5/train.log`.
- **Trainer container:** `sam2-trainer-cursor-dev-1` exited (code 255) about 2026-11-02. Nothing is training.
- **Next run: model v7** (James, 2026-11-20). Not started. There is no model v6: the model number was moved up to
  match its prompt encoding, `edge-confidence-v7`.
  - Depth-relative seed trust, carried circles and polygons, oversize augmentation, `circle_pull_through` retired.
  - Plans: `docs/adjacent_circle_seed_plan.md`, `docs/depth_relative_trust.md`.
  - Start it with `training.run_name=sam2_em_v7`. `training.run_name=auto` would choose `sam2_em_v6`, the next free
    number.
  - Training set v4 (below).

## Open decisions (James)

- **The 15% held back** (James, 2026-10-22). Checked 2026-11-20: nothing beyond validation is held back from training.
  `data.train_split` has been 0.9 since the first commit; v5 trained on 234,172 and validated on 27,959 (10.7%), plus
  one guard section on each side of every slab. The 25% -> 10% change was `val.epoch_subset`, a share of the
  validation split scored each epoch, so it never removed training samples. Training data v5 really never used: 75%
  of full tiles (`full_tile_keep_fraction` 0.25, the `TODO(v7)` in `data/fulltile.py`).
- **`circle_pull_through` retirement.** Should retirement wait until export link coverage is measured?
  Coverage looked low:
  - RPC1: 619 of 40,998 locations.
  - RPC2: 581 of 13,024 locations.

  The RPC1 figure may be low only because the v3 RPC1 `shapes.jsonl` is truncated (see Data). Measure it again
  after a re-export.
- **Serve-time prompt mismatches.** Should the five mismatches in `serve-train-prompt-parity.mdc` be fixed in the
  server, in the trainer, or both?
- **Open questions in `docs/mask_refinement_plan.md` §11:** τ, λ_leak, the merge threshold, the prompt mix, M_corr.
- ~~Viking proto proposal~~ settled 2026-10-25: James applied the design in Viking `ee14a0b0`. It is in
  `gRPC_Protos` `Common/geometry.proto` and `segmentation.proto`, plus `segmentation.rst`. The proposal is kept,
  marked superseded. The Sam2Trainer copy, `annotation-gallery/seg_proto/segmentation.proto`, follows it (updated
  2026-11-16).

## Set v4 (2026-11-20)

- `Y:\SAM2TrainingData\TEM\v4` (`/volume1/Data/SAM2TrainingData/TEM/v4` on Storage6, 192.168.0.199) is a btrfs
  reflink copy of set v3. Its file counts, sizes and manifests match v3's. The old in-folder `_work/` caches were moved to
  `v4/_work_cache/<VOL>`, where the current exporter looks for them.
- Not re-exported yet. Plan: `C:\Users\jande\.claude\plans\when-we-submit-annotations-compiled-rose.md`.
  - RC2, RPC1, RPC2: `-Update -RefreshOData -SetVersion v4 -ParentSet v3`.
  - RC1: a full `-Force` export from RC1_Crush's `TEM.Quantized` tileset, because the annotations were remapped to
    that mosaic.
- Set v3 stays as model v5 trained on it.
- Serving mismatch to raise with the server: when the segmentation server reads crushed tiles itself
  (`tile_source.py`), it stretches the raw codes per cell from the 1st to the 99th percentile. The exporter uses Viking's
  display window and gamma instead.

## Data (Nornir notes as of 2026-10-26; recheck)

- **Truncated `shapes.jsonl`.** The TEM v3 RC1 and RPC1 files stop at z 282 and z 742. In all, 2,652 and
  24,702 locations have no shape. They were exported before the 16,384-row OData paging fix.
- **Missing set versions.** Several sets have no `training_set.json`, or one with `set_version` null. SEM launch
  configs never pass `-SetVersion`.
- **Circle centres.** The exporter takes circle centres from the circumcentre of three ring points instead of `X`, `Y`.

## Resolved

- **Is the prompt encoder frozen?** Yes, and the README is right (checked 2026-11-20). `freeze_mode: none` trains
  everything except SAM2's `sam_prompt_encoder`; `unfreeze_mask_cnn: true` then unfreezes its `mask_downscaling`
  network (4,684 parameters). Frozen: the point, not-a-point and no-mask embeddings (1,536 parameters). The log's
  "224.4M / 224.4M (100%)" is rounding. `configure_trainable_params` is identical at v5's start (`fd3e381` plus its
  snapshot patch) and today, and all 18 v5 starts logged `[none]`. Resume refuses a changed freeze setting.
