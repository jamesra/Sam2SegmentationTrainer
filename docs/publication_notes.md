# Publication notes: Sam2Trainer

Fine-tuning SAM2 for cell segmentation in serial-section electron microscopy (TEM and SEM), with prompt families that
simulate how annotators prompt in Viking, and the prompt encoding shared with the segmentation server. Material for
future papers: citations, formulas, algorithmic decisions and the reasoning behind them. Index of all projects:
`~/.cursor/publication-notes.md`. Format and rules: the `publication-notes` skill.

## Methods and contributions

- **Depth-relative trust for seed prompts** (`docs/depth_relative_trust.md`). A seed's prompt value is scaled by a
  cosine profile of normalised depth, $\rho = 1 - d / D$, fully trusted in the inner half of the depth and zero in the
  outer 5%. The radius-relative circle trust is its special case ($\rho = \lVert p - c \rVert / R$), so one rule covers
  drawn circles, carried circles and carried polygons, and the circle diameter scale (0.95) is the profile's zero point.
- **Local-thickness normalisation for branching shapes** (same document, section 4). Normalising depth by the local
  radius $D_\ell(p)$, the radius of the largest inscribed disc containing $p$, instead of the shape's global maximum
  depth keeps a thin branch's own core instead of letting a large body's depth erase it. Its cores split a carried
  shape into islands; training prompts a target with the islands on it, production returns one mask per island, so a
  process that branches between sections is recovered branch by branch. Builds on Hildebrand and Rüegsegger (1997).
- **Carried prompts from adjacent sections** (`docs/adjacent_circle_seed_plan.md`). Circles and polygons linked to a
  target on a section 1-2 away are carried through the section registration and the published residual correction,
  the circle radius by a five-point mean scale, and used as prompts where they land when at least 60% of the carried
  shape lies on the target.
- **Prompt-size against object-size map** (`docs/depth_relative_trust.md`, section 9). The ratio of a prompt's depth to
  the selected object's depth, with a wrong-object rate, measures whether the model matches an object to the prompt's
  size and exposes annotators' oversized circles.
- **Edge-confidence prompt encoding (v6, served)** (`annotation_crops/prompt_encoding.py`,
  `docs/mask_refinement_plan.md` section 2): traced masks trusted in the interior and loosely at the edge; circle
  annotations as negatives by radius-relative trust; the max-or-unknown overlap rule.

## Formulas

All defined in `docs/depth_relative_trust.md`:

- Trust profile: $w(\rho) = 1$ for $\rho \le a$, $\tfrac12(1 + \cos(\pi (\rho - a)/(b - a)))$ for $a < \rho < b$, 0 for
  $\rho \ge b$; $a = 0.5$, $b = $ `circle_diameter_scale` $= 0.95$ (section 2).
- Global and local depth coordinates $\rho_g = 1 - d/D$, $\rho_\ell = 1 - d/D_\ell$,
  $D_\ell(p) = \max\{ r : \exists c,\ p \in B(c, r) \subseteq S \}$ (sections 3-4).
- Island overlap $o_i(P) = \sum_{I_i \cap P} w / \sum_{I_i} w$; carried-shape overlap $o = |X \cap T| / |X|$ (sections
  6, 8).
- Carry $\Phi = T_t^{-1} \circ C \circ T_s$, $C(v) = v + \alpha f_s(v) + \beta f_t(v)$; five-point radius (section 7).
- Size ratio $s = D_{\text{prompt}} / D_T$ (section 9).
- v6 edge confidence $c(d) = 1 - e^{-d/\tau_{\text{eff}}}$, $\tau_{\text{eff}} = \max(\tau_{\min}, \min(\tau, \kappa D))$
  (section 10).

## Algorithmic decisions

- **2026-10-25 Thin branches by local thickness, not by unioning with the target.** Rejected: rebuilding a missing
  branch prompt from the union of the carried shape and the target mask, because it puts the answer in the prompt
  (teaches copying) and production has no target mask.
- **2026-10-25 Islands: training uses the islands on the target; production keeps all.** A large lobed polygon can
  branch, and the target may be the smaller branch; production returns one mask per island.
- **2026-10-25 The excluded outer band is 5% of depth, not of area.** Depth makes the polygon rule reduce to the circle
  rule; 5% of area would not (the outer 0.05 R of a disc is 9.75% of its area).
- **2026-10-25 Retire `circle_pull_through`.** The synthetic circle across the target's edge stood in for a carried
  circle; real carried circles and polygons replace it (with the next run; removing a family changes the prompt
  layout). Its v5 record (training only; it has no validation family): drawn for 8.3% of variants (configured 5%),
  70,302 in epoch 5 with 237 (0.3%) falling back to `circle_seed`; the hardest family by training loss, median 0.50,
  0.30, 0.29, 0.26, 0.24, 0.23 over epochs 0-5 (TensorBoard `train/loss_by_prompt`, sampled points), 1.3-1.5x
  `circle_seed` throughout, about 1.1x `circle_annotator` after epoch 0. It was still improving, and it is v5's only
  training on circles that cross the cell's edge, so retiring it depends on the export giving carried circles to
  enough targets (v5 link coverage was low: 619 of 40,998 RPC1 and 581 of 13,024 RPC2 locations had a linked outline).
- **2026-10-25 A real carried circle always replaces the simulated circle seeds** for a target that has one passing the
  overlap rule; validation keeps the simulated families everywhere for comparability.
- **2026-10-25 Carried shapes need at least 60% of their area on the target**, the rule `adjacent_poly_seed` used; the
  export records every overlap and the trainer filters.
- **2026-10-25 Tapered foreground circle seeds (v7, pending the eval).** In training, seed circles were inscribed in the
  traced polygon (accurate radius) and annotator circles, whose radii are unreliable, were only negatives, so only the
  negatives were tapered (v6, Oct 15). In production the seed is the annotator's circle, often oversized; the eval-only
  comparison on v5.4 is `docs/circle_seed_trust_eval.md`.
- **2026-10-25 Oversize augmentation and `circle_oversize` validation.** The annotator-like jitter only draws circles
  smaller than the inscribed one (`scale_range` 0.45-1.0 of depth), so training never showed an oversized circle; about
  10% of circle seeds scaled 1.2-2x, kept modest so the size cue survives.

## Discussion points

- **The Oct 13 circle-size measurement understated circle size.** It compared each circle's radius with the equal-area
  radius of its linked polygon, which exceeds the inscribed radius for elongated cells; annotators often oversize
  circles, which the median ratio of 0.45 hid (`docs/training_plan_assessment.md`, 1.3).
- **Size matching is learned only as far as training shows it.** The fully trusted core says "at least this much is
  object", which favours a soma over a thin process under the same circle, but oversize augmentation and carried-circle
  noise weaken the size cue; measured by the size map and the wrong-object rate.
- **Adjacent-section prompts were starved by link coverage**, not by design: 0.7% realised of a 15% share in v5.
- **An inference-only test of a new encoding** on a model trained with the old one shows the model's tolerance, not
  what training with the new encoding gives.

## References

- Hildebrand, T., & Rüegsegger, P. (1997). A new method for the model-independent assessment of thickness in
  three-dimensional images. *Journal of Microscopy*, 185(1), 67-75. doi:10.1046/j.1365-2818.1997.1340694.x (verify).
  Defines local thickness: the diameter of the largest sphere (disc, in 2-D) that fits inside the structure and contains
  the point. Used for: the local radius $D_\ell(p)$ (half its local thickness) in the depth-relative trust of branching
  shapes. The same operator is ImageJ/Fiji's "Local Thickness" plugin.
- Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714. Used for: the model being
  fine-tuned.

## Open questions for a paper

- The plain-versus-tapered seed comparison on v5.4, then on a model trained with v7.
- Carried-circle and carried-polygon results by carried-centre depth and size ratio, once the export has them.
