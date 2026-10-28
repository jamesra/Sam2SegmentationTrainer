# Annotator-like circle seeds: `circle_annotator` validation and `prompt.circle_seed_jitter`

Oct 16, 2026 (revised 09:40 after annotator review) · Source: the Sam2 project docs *Mask refinement and edge
tolerance plan* (sections 3.7, 6, 8, 9 item 13) and *Training plan assessment* 1.3. The copies in this `docs/`
folder predate them; this file is self-contained.

## Why

Every circle seed the trainer builds today, in training and in validation, is the target's inscribed circle x 0.95,
centred exactly on the deepest pixel. Production seeds are circles drawn by annotators in Viking. Annotators place
a circle near the visual centre of a large part of the cell (for a horizontal cell, in one of its big lobes, not
necessarily the global deepest point), with a small placement error, and keep it inside the cell, usually somewhat
smaller than the largest circle that would fit there. Measured sizes (annotator radius / inscribed radius of the
linked polygon one section away) have median 0.87.

The first version of this rule (centres anywhere from 0.3 to 1.0 of the maximum depth, radius allowed 10 px past
the centre's depth) was rejected on review: no annotator misses the centre by that much. That spread describes a
circle carried forward from another section, which is not this family.

## The annotator-like circle (shared helper)

Add to `data/prompts.py`:

```python
def annotator_circle(target: np.ndarray, rng: np.random.Generator, cfg: CircleSeedJitter) -> np.ndarray | None:
    """Boolean disc for an annotator-like circle seed inside the boolean `target`, or None (caller keeps the
    inscribed seed)."""
```

1. `dist` = Euclidean distance transform of `target`, computed inside the target's bounding box padded with
   background by 1 px. `D = dist.max()`; return None if `D == 0`.
2. **Big areas.** Find local maxima of `dist` (`skimage.feature.peak_local_max` or an equivalent maximum filter)
   with `dist >= cfg.peak_min_frac * D`, separated by at least `max(3, 0.5 * dist[peak])` px. The global maximum is
   always a candidate. Pick one peak with probability proportional to `dist[peak]**2` (a bigger area is more likely
   to be chosen).
3. **Placement error.** Draw an offset uniformly within a disc of radius `cfg.offset_frac * dist[peak]` around the
   peak, rounded to a pixel. If that pixel is outside the target or has `dist < 0.5 * dist[peak]`, redraw (up to 8
   tries), else use the peak itself. Call the centre p.
4. **Size.** `r = dist[p] * U(cfg.radius_range)`. The circle is therefore always inside the target (r < dist[p]).
5. If `r < cfg.min_radius_px`, return None.
6. Return `rasterize_circle(shape, px + 0.5, py + 0.5, r)` (pixel centres, as `rasterize_circle` documents).

Draw order inside the helper is fixed: peak choice, then the offset (and any redraws), then the size. Keep it fixed;
tests pin it.

Config, new `prompt.circle_seed_jitter` block (a small frozen dataclass `CircleSeedJitter`, loaded and validated in
`PromptConfig.from_cfg` like `circle_annotation_trust`):

| Key | Default | Meaning |
| --- | --- | --- |
| `prob` | 0.7 | Share of training circle seeds replaced by an annotator-like circle. 0 disables and draws nothing |
| `peak_min_frac` | 0.5 | A "big area" is a local depth peak at least this fraction of the target's maximum depth |
| `offset_frac` | 0.15 | Centre placement error, as a fraction of the chosen peak's depth |
| `radius_range` | [0.55, 0.95] | Radius as a fraction of the depth at the chosen centre |
| `min_radius_px` | 6 | Smaller circles fall back to the inscribed seed |

Validate: `0 <= prob <= 1`, `0 < peak_min_frac <= 1`, `0 <= offset_frac < 0.5`,
`0 < radius_range[0] <= radius_range[1] < 1`, `min_radius_px >= 1`.

## Step 1: validation family `circle_annotator`

In `data/dataset.py`:

- `"circle_annotator"` is appended at the END of `VAL_PROMPTS`, so `VAL_FAMILY_CODES` of every existing family is
  unchanged and their validation prompts stay bit-identical.
- `VAL_MIXED_TYPES["circle_annotator"] = PromptType.CIRCLE_SEED`; it is in `CIRCLE_PROMPTS`; it keeps negatives.
- In `_families_item`, for this family only: build the disc with
  `annotator_circle(target_aug, np.random.default_rng([self.seed, sample_id, VAL_FAMILY_CODES["circle_annotator"], 2]), jitter_cfg)`
  (prob is ignored: always jitter). If it returns None, fall back to `circle_aug`. Pass that disc as `circle_mask` to
  `build_prompt`. Every other family keeps `circle_aug`.

In `validation.py` and `config/em_finetune.yaml`: `circle_annotator` is in `DEFAULT_VAL_PROMPTS`, in
`eval.val_prompts` (after `circle_mask_noneg`) and in `NEG_SPLIT_PROMPTS`.

## Step 2: training rule

> **Superseded Oct 18, 2026.** The per-variant replacement below (and `circle_seed_jitter.prob`) is gone. The circle
> prompts are now three peer prompt types in the `prob_*` mix: `circle_seed` (0.10, the plain circle),
> `circle_annotator` (0.20, the disc from `annotator_circle`, generator `[seed, epoch, index, k, 2]`) and
> `circle_pull_through` (0.05, see "Pull-through circles" below, generator `[seed, epoch, index, k, 3]`). No variant
> is forced to be a circle: all four variants are drawn from the mix without replacement. A disc that cannot be drawn
> falls back to the plain circle and the variant is reported as `circle_seed` (`requested_prompt_type` keeps what was
> drawn). An adjacent polygon seed that cannot be fitted falls back to the plain circle. A retired `prob` key in an
> old config's `circle_seed_jitter` block is ignored. The containment rule (inside the target only) stands.

In `_mixed_item`, after `circle_aug` is final (after the existing fallback that fits `inscribed_circle_xyr`), for each
variant `k` whose `kind` is `PromptType.CIRCLE_SEED` (variant 0 included):

```python
circle_k = circle_aug
if jitter.prob > 0:
    jrng = np.random.default_rng([self.seed, epoch, int(index), k, 2])
    if jrng.random() < jitter.prob:
        disc = annotator_circle(target_aug, jrng, jitter)
        if disc is not None:
            circle_k = disc
```

Pass `circle_k` as `circle_mask` to that variant's `build_prompt`. The adjacent-polygon-seed fallback to a circle also uses
`circle_k`. With `prob` 0 nothing is drawn, so prompts are bit-identical to the current code. Encoding, negatives
and losses are unchanged: no `prompt_encoding` version bump.

**Containment loss inside the target only.** Wherever the mask-containment weight is built from the seed's trust
(`prompt_fg`), multiply it by the ground-truth target, so the trusted seed core counts only where it overlaps the
target. With the rule above the disc is already inside the visible target before augmentation; this guards against
elastic warps and rasterization at the edge. The prompt itself is never clipped.

## Pull-through circles (`circle_pull_through`)

Added Oct 18, 2026. An annotator sometimes draws the circle across the cell edge. `pull_through_circle(target, rng,
cfg, depth=None, info=None)` in `data/prompts.py` draws such a disc: per try, the radius `U(radius_range) x D`
(D the target's maximum depth, raised to `min_radius_px`), then a centre among target pixels whose depth is between
0.25 and 0.9 of the radius; the first disc with at least `overlap_min` of its area inside the target and at least
`outside_min` outside is returned, else None after `max_tries`. The disc is not clipped; the containment weight is
zero outside the target, so the model learns to ignore that part and expand from the inside part to the full cell.

`prompt.circle_pull_through` (`CirclePullThrough`): `overlap_min` 0.60, `outside_min` 0.05, `radius_range`
[0.4, 0.9], `min_radius_px` 6, `max_tries` 20. Its share is `prompt.prob_circle_pull_through` (0.05). There is no
validation family for it yet (possible follow-up). Tests: `tests/test_circle_pull_through.py`. Gallery:
`docs/prompt_overview.md`.

## Tests (`tests/`)

- `annotator_circle` on synthetic targets (a disc, a 3:1 ellipse, a thin bar, a C shape, and a two-lobed shape with
  lobes of depth 100 and 60 px): the disc lies inside the target (allow 1 px of rasterization); the centre is within
  `offset_frac * dist[peak]` of a chosen peak whose depth is at least `peak_min_frac * D`; on the two-lobed shape both
  lobes are chosen, the deeper one more often (about 0.74 vs 0.26 for depths 100 and 60); the radius divided by
  `dist[p]` lies in `radius_range`; None for an empty target.
- Same `(seed, epoch, index, k)` gives the same disc; different `k` gives independent draws.
- `prob` 0: `_mixed_item` output identical to the current code for a fixed set of indices.
- Containment: a seed disc that extends past the target gives zero containment weight outside the target and
  unchanged weight inside it.
- Validation: existing families' prompts unchanged after adding `circle_annotator`; `circle_annotator` is identical
  across two epochs and two model loads.

## Not in this change

- Circles carried forward from another section (Viking copies a circle to the next section) are a different prompt:
  real cell drift moves them far off-centre. Measured Oct 16 against the next section's polygon: centre depth median
  0.57 (RPC1) and 0.36 (RPC2), radius / inscribed median 0.87. If they reach the model unadjusted in production they
  get their own family later; they are not part of `circle_annotator`.
- No change to `circle_diameter_scale`, the exporter, the server, or the prompt mix.
