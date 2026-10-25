# Circle-seed trust: an evaluation-only test

`docs/mask_refinement_plan.md` lists **radius-relative seed trust** as proposed, not decided. It is still not decided,
and training does not change. This page covers the two `sam2-em-eval --goal` options that test the idea on an
existing model. The code is in `sam2_segmentation_trainer/data/circle_seed_trust.py`, wired into the validation
families in `data/dataset.py`, `validation.py`, `train.build_validation_suite` and `evaluate.py`. The tests are in
`tests/test_circle_seed_trust.py` and `tests/test_circle_seed_trust_families.py`.

## Why

In training, a seed circle is the inscribed circle of a traced polygon (`circle_mask`), so its radius is accurate, or
an annotator-like circle (`circle_annotator`, `prompt.circle_seed_jitter`). The jitter only makes circles that are
smaller or off centre. Production seeds are circles drawn by annotators, and their radii are unreliable: annotators
often draw them too large, and some carry a 2x radius bug. Today a seed disc (true radius x
`circle_diameter_scale`, 0.95) goes into the prompt at the edge confidence `+seed_logit * c(d)`, which trusts
everything more than about 24 px inside the disc's edge. An oversized circle therefore tells the model, with full
trust, that the membrane and the cells beyond it are object.

## The options

| Option | Effect |
|--------|--------|
| `--circle-seed-taper FULL_FRAC` | Every circle-seed family (`circle_mask`, `circle_mask_noneg`, `circle_annotator`, `circle_oversize`) is fed as `+seed_logit * w(rho)` instead of `+seed_logit * c(d)`. `rho` is the distance from the circle's centre over its TRUE radius (before the 0.95). `w` is the cosine taper that circle annotations already use as negatives (`annotation_crops.prompt_encoding.circle_trust`): 1 out to `FULL_FRAC`, a half cosine down to 0 at `zero_frac`. `zero_frac` is the diameter scale the run resolved (the model's record, or `--circle-scale`), so the claimed region is the same disc as before. Try `0.5`. Off by default. |
| `--circle-oversize-range LOW,HIGH` | The factor range of the `circle_oversize` family (default `1.2,2.0`). |
| family `circle_oversize` | The target's inscribed circle, centre unchanged, with its true radius times a factor `U(LOW, HIGH)`, drawn like any seed circle (times the diameter scale). It reaches past the membrane, as an annotator's oversized circle does. Same target, same metrics and the same `with_neg` / `no_neg` split as `circle_mask`. Each sample's factor comes from its own generator `[training.seed, sample, 7, 2]`, so it is the same in every run and does not depend on which families run with it. It is never a default family and never part of the training mix; name it in `eval.val_prompts`. |

The negatives, the overlap rule (a pixel that a positive and a negative region both claim is 0), the centre click,
the background points and the area-average reduction to 256 x 256 do not change. Only the values inside the disc
change. With the taper absent, every prompt is bit-identical to before (a pinned test checks this). The prompt
encoding version (`edge-confidence-v6`), `encoding_metadata`, exported files and the server are not touched.
`goal_metrics.json` gets a `circle_seed` block (the taper and the oversize range) only when one of them is in use.

Each family's true radius:

* `circle_mask`, `circle_mask_noneg`: the tile's inscribed circle from `circle_annotations.json`, whose loaded radius
  is already scaled, so the true radius is that radius / scale. When the tile has none, it is the circle fitted to the
  mask (`inscribed_circle_xyr`) / scale.
* `circle_annotator`: the drawn disc's radius / scale, so that the taper covers the disc the default encoding feeds.
  The jitter draws that disc directly with no 0.95, so in this family "true radius" is a convention.
* `circle_oversize`: the inscribed true radius x the factor. Its full-trust core (`FULL_FRAC` x factor x R) reaches
  the inscribed circle's edge once the factor is `1 / FULL_FRAC` (2 at 0.5).

## Running the comparison

v5.4 is epoch 4 of run `sam2_em_v5` (`serve/serve_epoch4.pt`, with its `serve_epoch4.prompt_encoding.json`
beside it). Its validation set is the frozen split `v5`. Run both passes with the same split, data root and
families, one without the taper and one with it. Inside the trainer container (`/workspace` is this repo, `/outputs`
is `D:\Docker\mounted-configs\sam2-trainer` on the host, and `/storage4` is the NAS), run from `/workspace` so the
working tree's code is used:

```bash
cd /workspace
MODEL=/storage4/Sam2Trainer/runs/sam2_em_v5/serve/serve_epoch4.pt
FAMILIES='eval.val_prompts=[circle_mask,circle_annotator,circle_oversize]'
python -m sam2_segmentation_trainer.evaluate --goal --checkpoint "$MODEL" --split v5 --set "$FAMILIES" \
    --out /outputs/eval_v5.4_circle_seed_edge 2>&1 | tee /outputs/eval_v5.4_circle_seed_edge.log
python -m sam2_segmentation_trainer.evaluate --goal --checkpoint "$MODEL" --split v5 --set "$FAMILIES" \
    --circle-seed-taper 0.5 --out /outputs/eval_v5.4_circle_seed_taper 2>&1 | tee /outputs/eval_v5.4_circle_seed_taper.log
python scripts/compare_circle_seed_trust.py /outputs/eval_v5.4_circle_seed_edge/goal_metrics.json \
    /outputs/eval_v5.4_circle_seed_taper/goal_metrics.json
```

Add `--set data.gold_set=gold_v1` to both passes for the gold row, and `--circle-oversize-range 1.5,3.0` (to both)
for a harsher oversize test. The comparison script needs only the standard library, so it also runs on the host:

```powershell
python scripts\compare_circle_seed_trust.py D:\Docker\mounted-configs\sam2-trainer\eval_v5.4_circle_seed_edge\goal_metrics.json `
    D:\Docker\mounted-configs\sam2-trainer\eval_v5.4_circle_seed_taper\goal_metrics.json
```

For each family it prints core recall (pooled, p10, share at target), leak, merge rate, spill and fragmentation for
both runs and the difference, with `+` where the taper is better and `-` where it is worse, for all samples, for
`with_neg` / `no_neg`, and for gold when both runs have it.

**GPU memory.** The eval loads SAM2.1 Hiera-L (about 224 M parameters, about 0.9 GB in fp32) and validates in fp32
with no autocast. Each step encodes `val.batch_size` (4) images of 1024 x 1024 and decodes one prompt per image and
family. Expect several GB, roughly 6 to 12 GB at batch size 4 (an estimate, not a measurement). `--set
val.batch_size=1` lowers it to roughly a third and changes no result: prompts are seeded per sample, not per batch.
`--num-workers` only sets the CPU loaders. Do not run it on the GPU while a training run is using it.

## Reading the result

The model was trained on the edge-confidence seed, never on the taper. The test therefore measures how v5.4 reacts
to a differently encoded seed at inference: a fair test of whether the taper limits the harm of an oversized circle
(`circle_oversize`, and `circle_annotator`), and a check of how much it costs on accurate circles (`circle_mask`). A
gain here argues for training with the taper; a loss on `circle_mask` may only mean the model has not learned it. The
taper would need its own encoding version and server support before any model trained with it is served.

## Result: v5.4, 2026-10-25

Full v5 split (27,959 samples, 26,494 per circle family), `serve_epoch4.pt`, `val.batch_size=2`, plain disc (`off`)
against the tapered seed (`on`, `full_frac` 0.5, `zero_frac` 0.95). Output: `/outputs/eval_v5.4_circle_seed_edge`,
`/outputs/eval_v5.4_circle_seed_taper`, comparison `/outputs/eval_v5.4_circle_seed_compare.txt`.

| Family | IoU off / on | core recall off / on | at target off / on | spill off / on | merge off / on | frag off / on |
| --- | --- | --- | --- | --- | --- | --- |
| `circle_mask` | 0.895 / 0.895 | 0.979 / 0.972 | 70.7% / 64.8% | 0.032 / 0.025 | 0.02% / 0.02% | 0.061 / 0.057 |
| `circle_annotator` | 0.877 / 0.871 | 0.966 / 0.947 | 61.1% / 55.5% | 0.030 / 0.023 | 0.01% / 0.00% | 0.061 / 0.064 |
| `circle_oversize` | 0.763 / 0.797 | 0.977 / 0.979 | 80.9% / 78.8% | 0.166 / 0.131 | 0.56% / 0.48% | 0.173 / 0.116 |

The with_neg and no_neg splits move the same way. Reading: v5.4 was trained on full-strength discs, so the taper's
weaker rim reads as a smaller object and every mask shrinks. That removes a fifth of the spill and a third of the
fragmentation for oversized circles but costs core recall on accurate and small ones. The taper is not served on v5.4;
it is the training encoding for v7, where the model learns what the taper means (`docs/adjacent_circle_seed_plan.md`,
section 3).
