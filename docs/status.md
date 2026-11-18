# Sam2Trainer status

The current state, dated. Rules and specs say how things work; this file says where things stand. Check every entry
against the logs before acting on it (the `sam2trainer-startup` skill), and update it when it has changed.

## Runs (checked 2026-11-16)

- **Model v5** (`sam2_em_v5`, trained on training set v3, validated on validation set v5).
  - Finished 2026-10-31: early stopping at epoch 10.
  - Best epoch 7 (model v5.7): core recall 0.9732, IoU 0.8943, spill 0.026, merge rate 0.000, fragmentation 0.042.
  - Run directory: `/storage4/Sam2Trainer/runs/sam2_em_v5`; log at `/outputs/sam2_em_v5/train.log`.
- **Trainer container:** `sam2-trainer-cursor-dev-1` exited (code 255) about 2026-11-02. Nothing is training.
- **Next run: model v7.**
  - Depth-relative seed trust, carried circles and polygons, oversize augmentation, `circle_pull_through` retired.
  - Plans: `docs/adjacent_circle_seed_plan.md`, `docs/depth_relative_trust.md`.
  - "v7" is the encoding's name in those plans. Check the model number before starting the run.

## Open decisions (James)

- **The 15% held back.** Should the extra 15% of samples still be held back from training, now that 10% is the
  validation subset? Raised 2026-10-22.
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

## Data (Nornir notes as of 2026-10-26; recheck)

- **Truncated `shapes.jsonl`.** The TEM v3 RC1 and RPC1 files stop at z 282 and z 742. In all, 2,652 and
  24,702 locations have no shape. They were exported before the 16,384-row OData paging fix.
- **Missing set versions.** Several sets have no `training_set.json`, or one with `set_version` null. SEM launch
  configs never pass `-SetVersion`.
- **Circle centres.** The exporter takes circle centres from the circumcentre of three ring points instead of `X`, `Y`.

## Discrepancies to resolve

- **Is the prompt encoder frozen?** The README (lines 216 and 229) says it is frozen except `mask_downscaling`.
  The config says `freeze_mode: none`, and every v5 resume logged 100% trainable parameters (224.4M). Either the
  README or the config is out of date.

