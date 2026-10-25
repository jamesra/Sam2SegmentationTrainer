# Adjacent-section circle seed: plan

Status: plan, Oct 25 2026. Not implemented. Supersedes the "Circles carried forward from another section … get their own
family later" note in `circle_annotator_spec.md` ("Not in this change").

## Why

In production an annotator often marks a cell as a circle on one section and a polygon on the next. Viking can carry
the circle to the adjacent section: its centre goes through the section correction service, and auto-segment should
then segment the cell there from the carried circle. For non-terminal circle annotations that would add the
neighbouring sections' outlines automatically.

v5 never trained on that prompt:

- `circle_pull_through` (5%) is a synthetic circle drawn across the target's own edge, not a carried one, and it is a
  plain disc.
- `adjacent_poly_seed` (15% configured) carries the linked *polygon*, not a circle, and realised only 0.7% of v5's
  variants (epoch 4: 6,329 of 931,511) because few polygons in `shapes.jsonl` have a linked outline.
- Carried circles are hard for the current seed encoding: measured Oct 16 against the next section's polygon, the
  centre depth median is 0.57 (RPC1) and 0.36 (RPC2) and radius / inscribed is 0.87, so the rim of a carried circle
  often lies outside the cell. A full-strength disc tells the model that rim is object.

## Design summary

1. The exporter writes, per volume, every circle that is linked to a polygon within `LINK_MAX_DZ` (2) sections, already
   carried into the polygon's section: centre and radius in the polygon section's mosaic pixels.
2. The trainer adds an `adjacent_circle_seed` prompt family that uses those circles where they land, encoded with the
   radius-relative tapered seed trust (new prompt encoding, see below), and an `adjacent_circle` validation family.
3. The server builds the same tapered seed from a circle the client sends (proto change, to follow the user's
   instructions), and Viking's auto-segment carries non-terminal circles to adjacent sections.

## 1. Export (nornir `segmentationtraining`)

### What is exported

Only circles linked to a polygon, not every circle (about 2M circles exist; only the linked ones are prompts).

A new file `{Output}/adjacent_circles.jsonl`, one line per (circle, linked polygon) pair:

```json
{"circle_location_id": 101, "circle_z": 204, "polygon_location_id": 102, "polygon_z": 205,
 "structure_id": 77, "terminal": false,
 "source": {"center": [x, y], "radius": r},
 "carried": {"center": [x, y], "radius": r, "rim_mapped": 4},
 "transform": {"stos_group": "SliceToVolume1", "correction_set": "<manifest provenance id>", "convention": "+n-t"}}
```

- `source`: the circle in its own section's mosaic pixels at downsample 1, centre from the location's `X, Y` and the
  true radius (`Radius`; never scaled by the exporter, as for `circle_annotations.json`).
- `carried`: the same circle in the *polygon's* section mosaic pixels at downsample 1, so the trainer draws it in the
  polygon's crops with no transform of its own (the crop key gives the crop origin).
- `terminal`: the circle location's Viking terminal flag, so production-like subsets (non-terminal circles) can be
  selected; training uses every linked circle.
- One circle linked to two polygons (z - 1 and z + 1) gives two lines.
- Same structure only, `|dz| <= LINK_MAX_DZ`, both ends in the export.

### The carry, done at export

For each pair, map five points, the centre and the east, north, west and south rim points at the true radius
(Viking's circle rule, `.cursor/rules/circle-mapping-scale.mdc` in the Viking repo):

1. circle mosaic (`circle_z`) -> volume, through the circle section's slice-to-volume transform (the stos group the
   Viking volume and the correction service use, e.g. `SliceToVolume1`);
2. volume -> corrected volume, with the published residual correction set
   (`<corrections_root>/<VOL>/<stos group>/manifest.json`, `<z>.vfield.npz`) applied the way Viking applies it when it
   carries a location to the adjacent section;
3. corrected volume -> polygon mosaic (`polygon_z`), through the inverse of the polygon section's transform.

`carried.center` is the mapped centre. `carried.radius` is the true radius times the mean of
`|mapped rim - mapped centre| / radius` over the rim points that mapped (`rim_mapped` says how many). A pair whose centre
does not map, or whose rim maps for no point, is not written and is counted in the export report.

### Overlap rule: at least 60% of the carried circle on the polygon

A pair is written only when at least 60% (`OVERLAP_MIN`, the `adjacent_poly_seed` and `circle_pull_through` value) of
the carried disc's area lies inside the polygon. The disc is the carried circle at its carried TRUE radius (not scaled
by `circle_diameter_scale`), the polygon is the whole target outline in its own mosaic pixels with its holes, not a
crop of it, so the decision is the same for every crop of the cell. The exporter computes it exactly (shapely
intersection of the disc and the polygon, or a fine raster of the disc's bounding box) and records it as
`carried.overlap` on each line:

```json
"carried": {"center": [x, y], "radius": r, "rim_mapped": 4, "overlap": 0.83}
```

Pairs below the threshold are excluded from the file and counted in the export report, with their overlap
distribution, so the share a carry loses to drift is visible per volume.

The trainer's `data/projection.py` already implements steps 1-3 for outlines (grid stos only; groups where every section
maps straight to one reference section, so RC2, RPC1, RPC2 `SliceToVolume1`; RC1's chained `grid16` is reported and
skipped). The exporter should reuse that logic rather than re-derive it: either move `projection.py` into a shared
package both use, or port it to nornir using `nornir_imageregistration.files.stosfile` for the stos and a small reader
for the published vfields (nornir has none today). One implementation, so the trainer's adjacent polygons and the
exporter's carried circles cannot disagree.

**The correction must match production.** `projection.py` chose its residual convention (`v + a*f_neighbour + b*f_target`)
empirically on linked pairs (`mask_refinement_plan.md`, section 3.1). The export must use whatever Viking's carry uses
(the correction service's own carry, `GrpcSectionCorrectionService`), and the export report must show, on a sample of
pairs, that the exported carried centre equals the one Viking computes for the same circle.

### Links

`shapes.py` builds `linked` only between polygon locations it exports, so a circle-polygon link is dropped today. The
`LocationLinks` pull (cached in `location_links.json`) must cover circle locations too, and the circle side of each
link is resolved against the ingested circle records (`circles.py`, `load_circle_records`).

While there: find out why most polygons in `shapes.jsonl` have `"linked": []` (v5's `adjacent_poly_seed` realised 0.7%
of its 15% share). If links are being lost (filtering, structure mismatch, dz limit, the cache), fix it in the same pass.

### Export report

Per volume: circle-polygon pairs found, written, skipped (no map, unsupported stos group, missing correction set,
overlap below 60%), the overlap distribution of written and excluded pairs, the carried-centre distance to the
polygon's centroid and the carried centre's depth in the polygon (median, quartiles), radius / inscribed-radius
distribution, and the production-equality check above.

### Tests (nornir)

Five-point carry and mean-scale radius on a synthetic grid transform and vfield; a pair whose rim partly fails to map;
the overlap rule (a disc at 0.59 and at 0.61 of its area inside, a polygon with a hole under the disc); link resolution
circle <-> polygon (same structure, dz limit, both directions); the file is rebuilt by `-Update` without restitching
crops, like `shapes.jsonl`.

## 2. Trainer

### Prompt family `adjacent_circle_seed`

- Loaded per volume by `build_tile_context` from `adjacent_circles.jsonl`; no projection at load time (the file is
  already in the target's mosaic pixels).
- Drawn for a sample whose target polygon has at least one carried circle (generator `[seed, epoch, index, k, 6]`,
  one circle when there are several). Elsewhere its share moves to the other circle families, like
  `adjacent_poly_seed`.
- The circle is used where it lands, at its carried radius, not clipped and not re-centred: drift is the point.
- Only pairs with at least 60% of the carried disc on the polygon reach the trainer (the export's overlap rule). The
  dataset checks `carried.overlap >= prompt.adjacent_circle.overlap_min` (0.60) again on load, so an older or
  hand-made file cannot slip a looser pair in; a pair that fails is dropped and counted in
  `train/prompt_fallback/adjacent_circle_seed`. The rule is on the whole cell, so a circle that passes is used in every
  crop of it, even one where little of the disc falls in the tile.
- Encoded with the tapered seed trust (section 3); plus the circle's centre as a foreground point when the centre lies
  inside the cell, as for the other circle seeds.
- Starting share 10-15%, taken from `circle_seed` and `circle_annotator`; check the realised share in the epoch log.

### Validation family `adjacent_circle`

Real carried circles, scored like `circle_mask` (core recall, leak, merge rate, spill, fragmentation), split by
volume and by carried-centre depth, so the production workflow has its own curve. Kept separate from the synthetic
`circle_pull_through`.

## 3. Prompt encoding v7: tapered circle seeds

Today a circle seed is a disc encoded with the edge curve `+SEED_LOGIT * c(d)`. v7 encodes every foreground circle
seed (`circle_seed`, `circle_annotator`, `circle_pull_through`, `adjacent_circle_seed`) by its radius instead:
`+SEED_LOGIT * w(rho)`, `rho = |p - c| / R` with `R` the TRUE radius, `w = 1` for `rho <= full_frac` (0.5) and a
half-cosine to 0 at `zero_frac = circle_diameter_scale` (0.95): `circle_trust()` used positively. Negatives, the
overlap rule and the area-average reduction are unchanged. Polygon seeds, corrections and painted masks keep `c(d)`.

- New `PROMPT_ENCODING_VERSION` (`edge-confidence-v7`), recorded in the export metadata with the seed trust's
  `full_frac` and `zero_frac`; the server implements v7 beside v6 and picks by the model's manifest, so v5.4 keeps v6.
- With it, the proposed oversize augmentation (`mask_refinement_plan.md`, "Radius-relative seed trust"): about 10% of
  training circle seeds scaled 1.2-2x, since annotators often oversize circles, and the `circle_oversize` validation
  family.
- Evidence first: the eval-only comparison on v5.4 (`docs/circle_seed_trust_eval.md`: plain vs tapered seed on
  `circle_mask`, `circle_annotator`, `circle_oversize`) decides `full_frac` and whether v7 goes ahead.

## 4. Server, proto, Viking

- Proto: the client sends a circle seed as a circle (centre and true radius), so the server can build `w(rho)`; the
  exact message comes from the user's proto instructions. Clients stop rasterizing circle seeds and never scale them.
- Server: v7 encoder for circle seeds; v6 path unchanged for models that declare it.
- Viking: auto-segment carries non-terminal circles to the adjacent section (centre through the correction service,
  radius by the five-point mean scale) and requests a segmentation from the carried circle, once a v7 model is served.

## Order

1. Eval-only taper comparison on v5.4 (in progress) -> decide v7 parameters.
2. Exporter: links that include circles, `adjacent_circles.jsonl`, production-equality check, report.
3. Trainer: v7 encoding, `adjacent_circle_seed`, oversize augmentation, `adjacent_circle` and `circle_oversize`
   validation, overview images (`scripts/make_prompt_overview.py --families adjacent_circle_seed`).
4. Train the next model on v7.
5. Proto, server v7 encoder, Viking clients; then the carry in auto-segment.

## Open questions

- Which component does Viking's production carry use for the correction (the client's own field read, or the
  correction service's carry RPC), and with which convention? The export must match it exactly.
- RC1's chained stos group: support chained groups, or leave RC1 out of this family.
- The family's share, once the export report shows how many pairs pass the 60% overlap rule.
- Whether a carried circle should also be offered without its centre point when the centre lands outside the cell.
