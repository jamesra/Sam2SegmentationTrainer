# Planning documents

These are copies of the planning documents kept in the Sam2 claude.ai project, so agents working in this repository can read them. Where a document refers to another one by its project name, this table gives the file.

| Project document | File | What it covers |
| --- | --- | --- |
| *Overall training plan* | `overall_training_plan.md` | Goals, the exporter's circle file, the trainer data pipeline, the prompt mix, model and loss changes, the gallery, and the configuration |
| *claude/Mask refinement and edge tolerance plan* | `mask_refinement_plan.md` | Edge-tolerant prompts and loss, the refinement prompt families, goal-driven validation, and the phased implementation order. The working plan for the trainer, kept in step with the code |
| *claude/Training plan assessment* | `training_plan_assessment.md` | Review of the overall plan: weaknesses, decisions (z-slab split, circle scale, negative-only removal, fully masked tiles) and measurements |
| *claude/SAM3 evaluation* | `sam3_evaluation.md` | Whether to move from SAM 2 to SAM 3 |
| *claude/uSAM evaluation* | `usam_evaluation.md` | Whether to move from SAM 2 to μSAM |

`SAM2 Fine-Tuning Plan for TEM SEM Electron Microscopy.md` is the original implementation brief from September. The documents above supersede it where they differ.

When the plans change, update the project and these copies together.
