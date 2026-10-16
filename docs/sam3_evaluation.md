# SAM 3 Evaluation: Should We Move from SAM 2?

Oct 12, 2026 · Split out of the training plan assessment on Oct 13, 2026 · Context: SAM2.1 Hiera-L fine-tuned for interactive cell-membrane segmentation of serial-section TEM of retina (2.18 nm/px, 1024×1024 grayscale tiles); phase 2 is subcellular features, and a secondary goal is SEM.

The μSAM question is in *claude/uSAM evaluation*.

## Summary

I don't recommend switching the membrane pipeline now:
- For point, box and mask prompts on single images, SAM 3 beats SAM2.1-L by only about 1 mIoU, and runs at about half the speed.
- It has no official training code for the interactive path.
- It comes under a more restrictive licence.

It is worth evaluating for three things: phase 2 organelles found from exemplars, propagating masks across serial sections, and SEM.

## What SAM 3 is

- **Paper.** "SAM 3: Segment Anything with Concepts" (Carion et al., ICLR 2026), released Nov 2025; SAM 3.1 followed in Mar 2026.
- **Size.** About 848M parameters:
  - a roughly 450M Perception Encoder ViT (1008 px input, patch 14, single scale, with a ViTDet-style neck that synthesizes high-res maps);
  - a roughly 300M text encoder;
  - the detector plus tracker.
- **New capability.** *Promptable concept segmentation*: a text phrase or an image exemplar returns every instance of that concept.
- **SAM 2-style prompts.** It keeps point, box and mask prompts through `SAM3InteractiveImagePredictor`, which accepts `mask_input`.

## Evidence relevant to us

| Question | Finding | Source |
| --- | --- | --- |
| Interactive image accuracy vs SAM2.1-L | SA-37 mIoU at 1/3/5 clicks: 66.4/80.3/84.3 (SAM2.1-L) vs 66.1/81.3/85.1 (SAM 3). About +1 point, at 43.5 vs 93 FPS | SAM 3 paper, Table 7 |
| Video and tracking | Larger gains: MOSEv2 J&F 47.9 → 60.3; SA-V 78.4 → 84.4 | SAM 3 paper, Table 6 |
| Fine-tuning code | Official training covers only the detector/concept path. No official training for the tracker or interactive path | `README_TRAIN.md`; GitHub issues #318, #217 |
| Licence | SAM 2: Apache 2.0. SAM 3: custom SAM License (derivative weights must carry it, acknowledgment required, terms can change). Checkpoints are gated on Hugging Face | Repository LICENSE files |
| Cells, zero-shot | Text prompts trail μSAM badly on cell and nuclei instance segmentation (0.27 vs 0.48; 0.26 vs 0.51) and are prompt-sensitive | Archit & Pape, *MIDL* 2026 |
| EM organelles | POT-SAM3 keeps SAM 3 frozen and learns concept tokens from one annotated section of serial-section EM: Dice 0.46 → 0.77 on organelle datasets | MICCAI 2026 |
| EM membranes or neurites | No published evaluation found | — |

## What we would gain and lose

- **Membrane segmentation (phase 1): little.** Interactive accuracy is about the same as SAM2.1-L.
  - Membrane precision may be no better: SAM 3 uses a single-scale ViT with synthesized high-res features, while Hiera has a true stride-4/8 hierarchy.
  - Switching means porting `encode_images`/`decode_prompts`, the signed-mask pipeline and the gallery to new internals, and writing interactive-path training ourselves.
  - The model is about 4× larger than Hiera-L (224M), so batch 4 at 1024² on 32 GB would need a rework.
- **Subcellular features (phase 2): possibly a lot.** "Show one mitochondrion, ribbon or synapse, get all of them" is exactly SAM 3's new capability. POT-SAM3 is the closest peer-reviewed evidence that it transfers to serial-section EM.
- **Serial-section propagation: worth testing.** Treating z as time and propagating a finished section's masks to the next is where SAM 3's tracker gains are largest. This also works with SAM 2's video predictor, so it can be prototyped without switching.
- **SEM: plausible.** The POT-SAM3 datasets are FIB-SEM and SBF-SEM organelle data, which is relevant to the secondary goal.

## Recommendation

1. **First, build the held-out benchmark** described in *claude/Training plan assessment* (items 1.1–1.2): a held-out z-slab or 3D-block split per volume, circle prompts, NoC@90 and boundary F1. Without it, no model comparison is trustworthy.
2. **Then run a time-boxed bake-off** (about 1–2 weeks) on that benchmark.
   - Compare zero-shot SAM2.1-L, zero-shot SAM 3 interactive and our fine-tuned SAM 2, using identical circle and point prompts. *claude/uSAM evaluation* adds μSAM arms.
   - Move the membrane pipeline only if SAM 3 zero-shot is clearly better than SAM2.1-L zero-shot, say by more than 3 IoU points or one fewer click at NoC@90.
   - Even then, weigh the porting cost against the licence change.
3. **Prototype now, without switching:** SAM 3 exemplar prompting for organelles and SEM as the start of phase 2, and SAM 2 video-mode propagation across sections.

## References

- Carion N. et al. SAM 3: Segment Anything with Concepts. *ICLR* 2026. arXiv:2511.16719. Code: github.com/facebookresearch/sam3.
- Ravi N. et al. SAM 2: Segment Anything in Images and Videos. *ICLR* 2025. arXiv:2408.00714.
- Archit A. et al. Segment Anything for Microscopy. *Nat. Methods* 22, 579–591 (2025).
- POT-SAM3, *MICCAI* 2026: papers.miccai.org/miccai-2026/0802-Paper0801.html
- Archit A., Pape C. Revisiting foundation models for cell instance segmentation. *MIDL* 2026 (PMLR 324). arXiv:2603.17845.
