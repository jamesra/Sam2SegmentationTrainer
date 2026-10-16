# μSAM Evaluation: Should We Move from SAM 2?

Oct 13, 2026 · Companion to *claude/SAM3 evaluation* · Context: SAM2.1 Hiera-L fine-tuned for interactive cell-membrane segmentation of serial-section TEM of retina (2.18 nm/px, 1024×1024 grayscale tiles); phase 2 is subcellular features, and a secondary goal is SEM.

## Summary

I don't recommend switching to μSAM:
- **It is SAM 1 underneath.** Moving to it means a step back to SAM 1's single-scale ViT, which lacks SAM 2's stride-4/8 high-resolution features and runs about 6× slower.
- **No μSAM model covers our task.** Its EM models cover mitochondria, nuclei and other roundish organelles. The authors report their EM generalist gives "only marginal benefits or is detrimental" for neurites, and suggest starting from the default SAM model for them.
- **SAM 2 is not supported.** SAM 2 and SAM 3 support is still a roadmap item (docs, v1.8.14, Sep 2026).
- **Our pipeline is SAM 2-specific.** Circle prompts, ±10 signed masks and mixed-prompt decoding are built around SAM 2's internals, and μSAM's trainer supports none of them.

μSAM's value to us is as **a recipe and a set of baselines, not a backbone**:
- its iterative corrective-click training (already recommended in *claude/Training plan assessment* 1.4);
- its automatic instance-segmentation decoder;
- its tiling;
- and `vit_l_em_organelles` as a cheap zero-shot baseline for phase 2 mitochondria and for SEM.

## What μSAM is

- **Paper.** "Segment Anything for Microscopy" (Archit et al., *Nat. Methods* 22, 579–591, 2025). Code: `computational-cell-analytics/micro-sam`, MIT licence. Actively maintained: v1.8.14, Sep 2026.
- **Backbone.** The original SAM (ViT-B/L/H) plus MobileSAM's ViT-T. Input is resized to 1024², and grayscale is copied to 3 channels. The authors recommend ViT-B or ViT-L, finding no gain from ViT-H.
- **Fine-tuned models:**
  - light microscopy: `vit_*_lm`;
  - EM: `vit_*_em_organelles`, trained on MitoLab and MitoEM mitochondria plus PlatyEM nuclei;
  - histopathology and medical imaging variants.
  - The only EM model for cells and neurites, v1 `vit_b_em_boundaries`, is deprecated.
- **Training recipe.** Iterative prompting (after Kirillov et al.):
  - each step adds 8 rounds of corrective positive or negative points sampled from the error region;
  - the previous mask is fed back only 50% of the time, to avoid over-reliance;
  - losses are Dice for the mask and L2 for the IoU head;
  - Adam, LR 1e-5, batch 2, 512² patches.
- **AIS decoder.** A UNETR decoder on the image encoder predicts foreground, center distance and boundary distance. A seeded watershed turns those into instances. APG (MIDL 2026) turns the same outputs into point prompts for SAM.
- **Tooling:**
  - napari annotators for 2D, 3D and tracking;
  - precomputed and tiled embeddings (recommended tile 384, halo 64);
  - about 30 ms per prompt on CPU once the embedding exists;
  - 3D works by projecting a slice's mask to the neighboring slices as a prompt and re-running SAM.

## Evidence relevant to us

| Question | Finding | Source |
| --- | --- | --- |
| Backbone: SAM 1 vs SAM 2 on images | SA-23 mIoU at 1/5 clicks: SAM 58.1/81.3 at 21.7 FPS. SAM 2 trained on SA-1B only: 58.9/81.7. SAM 2 on its full data mix: 61.9/83.5. SAM 2 runs at 130.1 FPS. Equal or better accuracy at about 6× the speed | SAM 2 paper, Table 5 |
| Fine detail | SAM 2 adds Hiera's stride-4 and stride-8 features to the decoder's upsampling layers. SAM 1 decodes from a single stride-16 map. At 2.18 nm/px, a membrane is only a few pixels wide | SAM 2 paper, §4; architecture, not a measured EM result |
| μSAM on EM organelles | "Clear improvement for interactive segmentation due to fine-tuning for all datasets" (mitochondria, nuclei). AIS is about on par with MitoNet for automatic mitochondria | Archit et al. 2025 |
| μSAM on EM neurites and cells | The EM generalist gives "only marginal benefits or is detrimental" for neurites and ER; the authors advise starting from default SAM. No current EM model for cells or membranes | Archit et al. 2025; micro-sam docs |
| Automatic segmentation, SAM 2 vs μSAM | SAM 2 with automatic mask generation was "inferior to SAM" and "did not segment any objects for several datasets". μSAM AIS and APG are among the best on 36 LM and histopathology datasets (no EM tested) | Archit & Pape, *MIDL* 2026 |
| SAM 2 support in μSAM | "Support for SAM2 and SAM3" is listed as future work | micro-sam docs, Sep 2026 |
| Training cost | μSAM generalist: 250k iterations, about 6 days on one 80 GB A100. Our run: Hiera-L at batch 4, 1024², on 32 GB. SAM ViT-L's encoder alone (about 308M parameters) is larger than all of SAM2.1-L (224M) | Archit et al. 2025; SAM and SAM 2 model cards |
| Serial-section propagation | μSAM projects masks slice by slice. SAM 2 has a learned memory across frames. Recent EM-volume work builds on SAM 2's memory (Spatial-SAM, *CVPR* 2026; SAM4EM, *CVPRW* 2025); neither compares with μSAM | Papers cited |
| Licence | μSAM is MIT, SAM is Apache 2.0, and SAM 2 is Apache 2.0. Licence is not a factor, unlike SAM 3 | Repository LICENSE files |
| Retinal TEM membranes, μSAM vs SAM 2 | No published head-to-head found | — |

## What we would gain and lose

- **Membrane segmentation (phase 1): a net loss.**
  - We would trade Hiera-L for SAM 1 ViT-L: no high-resolution skip features, about 6× slower inference, and a larger encoder to fit at batch 4 on 32 GB.
  - There is no pretrained μSAM model to start from: the authors' own advice for neurites is default SAM, so we would fine-tune SAM 1 from scratch on our data.
  - μSAM's trainer would need our circle prompts, signed ±10 masks, negative context and containment loss added. The signed-mask idea ports, because SAM 1 has the same 256² `mask_input` and `mask_downscaling` path, but the rest is a rewrite.
- **Subcellular features (phase 2): a cheap head start for mitochondria only.**
  - `vit_l_em_organelles` with AIS gives zero-shot interactive and automatic mitochondria and nuclei segmentation.
  - Ribbons, synaptic vesicles, ER and synapses are not covered. SAM 3 exemplar prompting (see *claude/SAM3 evaluation*) is the more general route there.
- **Serial-section propagation: SAM 2 is the better base.** μSAM's slice-to-slice projection is roughly what SAM 2's video predictor does with a learned memory.
- **SEM: a plausible baseline.** The EM organelle model was trained largely on SEM-family volume data (MitoEM, MitoLab), so it may transfer to our SEM organelles better than our TEM-trained model. It will not help with SEM cell membranes.
- **Annotation tooling: useful outside Viking.** The napari annotators could be used to make phase 2 organelle ground truth, if a polygon import into Viking is written. They do not replace the Viking and server workflow for cells.

## What to borrow without switching

1. **Iterative corrective-click training.** This is the replacement for the full-mask prompt type proposed in *claude/Training plan assessment* 1.4.
   - μSAM's MIT-licensed implementation (`micro_sam.training`, on `torch_em`) is a tested reference: error-region click sampling, 8 rounds, previous logits fed 50% of the time.
   - Port the logic into `Sam2Trainer`, not the code.
2. **An AIS-style head for automatic proposals.**
   - A UNETR-type decoder predicting foreground plus center and boundary distances could sit on our fine-tuned Hiera encoder.
   - It would give "segment everything" proposals for a section, or prompts through APG.
   - This is a phase 2 or later idea, and untested on Hiera.
3. **Tiling with a halo** for embeddings of large fields. It is relevant to the field-of-view problem in *claude/Training plan assessment* 1.8, alongside the D2/D4 tiles.
4. **Evaluation code.** `run_evaluation_for_iterative_prompting` and the mSA metric are a ready-made check for the NoC@90 benchmark.

## Recommendation

1. **Stay on SAM 2 for the membrane pipeline.** Nothing in the evidence suggests μSAM would segment retinal membranes better, and the backbone would be a step back.
2. **Add μSAM arms to the bake-off planned in *claude/SAM3 evaluation*.** Use the same held-out z-slab benchmark, circle and point prompts, NoC@90 and boundary F1.
   - Membranes: zero-shot μSAM `vit_l` (default SAM ViT-L) as a SAM 1 reference beside zero-shot SAM2.1-L, SAM 3 and our fine-tuned SAM 2.
   - Phase 2: `vit_l_em_organelles` (interactive and AIS) on a small retinal mitochondria set, and on SEM if a test set exists.
   - Install it in its own environment, not the trainer's venv.
3. **Apply the same switching bar as for SAM 3.** Revisit only if μSAM beats SAM2.1-L zero-shot by more than 3 IoU points or one fewer click at NoC@90. Also revisit if μSAM releases SAM 2-based EM models, which would remove the backbone objection.
4. **Port iterative corrective training** from μSAM's recipe into our trainer as the next training-side change. It is the part of μSAM most likely to reduce annotator clicks.

## References

- Archit A. et al. Segment Anything for Microscopy. *Nat. Methods* 22, 579–591 (2025). doi:10.1038/s41592-024-02580-4.
- Archit A., Pape C. Revisiting foundation models for cell instance segmentation. *MIDL* 2026 (PMLR 324). arXiv:2603.17845.
- micro-sam documentation and releases: computational-cell-analytics.github.io/micro-sam; github.com/computational-cell-analytics/micro-sam.
- Ravi N. et al. SAM 2: Segment Anything in Images and Videos. *ICLR* 2025. arXiv:2408.00714.
- Kirillov A. et al. Segment Anything. *ICCV* 2023.
- Huang et al. Spatial-SAM: Spatially Consistent 3D Electron Microscopy Segmentation with SDF Memory and Semi-Supervised Learning. *CVPR* 2026.
- Shah et al. SAM4EM: Efficient memory-based two stage prompt-free segment anything model adapter for complex 3D neuroscience electron microscopy stacks. *CVPRW* 2025.
