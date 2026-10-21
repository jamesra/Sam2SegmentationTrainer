# Validation sets: frozen splits and gold sets

Validation scores are only comparable between runs that validate on the same locations. A run normally builds its
own split from the data it sees, so a new data revision moves the validation slabs: v5's validation set shares only
2,711 of its 8,251 locations with v4's, and 63% of v5's validation locations were training locations for v3 (whose
split was the older random location split). Two files fix that. Both live under
`{SAM2_CHECKPOINT_ROOT}/valsets` (`/storage4/Sam2Trainer/valsets`) and are made with `sam2-em-valsets`
(`python -m sam2_segmentation_trainer.valsets` before the package is reinstalled).

| | File | What it is | Used by |
|---|---|---|---|
| Frozen split | `valsets/splits/<name>.json` | A run's `split.json` saved under a name | `data.frozen_split=<name>` |
| Gold set | `valsets/gold/<name>.json` | Reviewed location ids per volume, drawn from a frozen split's validation locations | `data.gold_set=<name>` |

Validation sets are versioned like training sets: a frozen split is `v<N>`, and a revision of one is
`v<N>.<R>` (`v5`, `v5.1`). The sequence is its own: validation set v5 is the split of the v5 run (which trained on
training set v3), and its number says nothing about training sets or models. Say "validation set v5" for a frozen
split and "gold set gold_v1" for a gold set. Names are letters, digits, `_`, `.` and `-`.

| Validation set | Frozen from | Validation locations | Frozen |
|---|---|---|---|
| `v5` | `sam2_em_v5` (training set v3, z-slab split) | 8,251: RC1 2,359, RC2 517, RPC1 3,585, RPC2 1,790 | 2026-10-20 |

## Frozen split

```
sam2-em-valsets freeze --run sam2_em_v5 --name v5 --description "v5 run's z-slab split (training set v3)"
```

The file is the run's `split.json` plus a `frozen` block (name, source run, time, validation locations per volume).
It is never rewritten. A **new** run started with `data.frozen_split=v5` copies it to its own `split.json` and
adds the locations that appeared since: in a `zslab` split each is placed by its section (validation slab, guard
band, or training), so later data revisions never cross the split. A run that already has a `split.json` keeps it.
A volume with no validation sections in the frozen split (a volume added later) is trained on but not validated,
and the run warns.

Prefer a `zslab` split. A `location` split puts validation locations beside training locations in adjacent
sections, which inflates its scores; `freeze` warns.

## Gold set

```
sam2-em-valsets sample --split v5 --name gold_v1 --per-volume 100 --root /data-local/v3
```

`sample` draws `--per-volume` locations at random (seeded by `--seed` and the volume) from the split's validation
locations that exist in the data now, leaving out ignored locations and tiles of the wrong size. Random draws are
what make the set representative; do not hand-pick from flagged or high-loss locations. It writes a draft gold file
and one review list per volume (`valsets/gold/gold_v1.review/<volume>.txt`).

**Review** in the gallery, one volume at a time: paste the volume's list into the location filter, then approve
each location that is correct (approving the whole location is simplest) and reject each that is wrong. The gallery
writes `approved.json` and `ignore.json` as usual.

```
sam2-em-valsets build --name gold_v1               # read the review; repeat as often as needed
sam2-em-valsets build --name gold_v1 --finalize    # once nothing is pending
sam2-em-valsets check --name gold_v1               # every gold location is a validation location
sam2-em-valsets list
```

`build` sorts each candidate:

* **gold**: the whole location is approved, or every one of its current crop windows is;
* **rejected**: the location is in `ignore.json`, or it has no crop windows left in the data;
* **pending**: anything else (not reviewed, or only some windows approved).

`--finalize` refuses while candidates are pending (`--allow-pending` leaves them out). A final gold set never
changes; to revise it, sample or copy into a new name such as `gold_v1.1`.

Rejected candidates are not replaced, so a volume with many rejections ends below `--per-volume`. As a rough guide,
if per-location IoU varies by about ±0.1, 100 locations per volume (about 400 in all) measure the gold-set mean to
about ±0.01, enough to separate runs; 24 locations only to about ±0.04.

## In a run

`data.gold_set=gold_v1`:

* gold locations are never trained on (a gold location found in the training partition is dropped from training
  with a warning; that only happens when the gold set was drawn from a different split);
* every window of every gold location joins the per-epoch validation subset, like Approved masks;
* their scores are reported as their own row: `val_subset_gold/<metric>/<prompt>` (the goal metrics),
  `val_subset_gold/iou/<prompt>`, `val_subset_gold/n/<prompt>` and `val_subset_gold_n/<prompt>/<volume>`, and the
  same under `val_full_gold` on full passes; the transcript summary adds `gold (n=...)`.

Checkpoint selection still uses the main validation row. A draft gold set works but warns, since its rows can change
between runs until it is finalized. `inputs.json` records `frozen_split` and `gold_set` when they are set.

## Scoring an older checkpoint the same way

```
sam2-em-eval --goal --checkpoint <v3 weights> --split v5 --set data.gold_set=gold_v1 --out /outputs/eval_v3_valset_v5
```

`--split` takes a frozen split's name. Note that a checkpoint trained on a different split may have trained on some
of the frozen split's validation locations (v3 trained on most of v5's), which flatters it.
