# Publication notes

Material for future papers on the Sam2Trainer / Viking segmentation work: what is new, where it is written down, and
what it builds on. Add to it as methods are settled; keep the full references here so a paper can be drafted from
this file.

## Methods to write up

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

## References

- Hildebrand, T., & Rüegsegger, P. (1997). A new method for the model-independent assessment of thickness in
  three-dimensional images. *Journal of Microscopy*, 185(1), 67-75. doi:10.1046/j.1365-2818.1997.1340694.x.
  Defines local thickness: the diameter of the largest sphere (disc, in 2-D) that fits inside the structure and contains
  the point. Used for the local radius $D_\ell(p)$ (half its local thickness) in the depth-relative trust of branching
  shapes. The same operator is ImageJ/Fiji's "Local Thickness" plugin.
