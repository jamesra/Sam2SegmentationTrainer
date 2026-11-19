---
name: sam2trainer-startup
description: Get oriented at the start of a Sam2Trainer session, or before touching a training run - where the code, data, runs, logs and specs are, how to check what is running, how to pause, resume and evaluate a run safely, and which doc answers which question. Use when a session starts in Sam2Trainer, when asked about a run's state, or before pausing, resuming, evaluating or exporting a model.
---

# Sam2Trainer startup

## 1. Read the current state first

`docs/status.md` holds the dated state: the latest run, what is served, open decisions and uncommitted work. Then
check that it is still true, because rules and docs go stale while a run moves on:

```powershell
docker ps -a --format '{{.Names}}\t{{.Status}}' | Select-String 'sam2|segmentation|correction'
Get-Content D:\Docker\mounted-configs\sam2-trainer\<run>\train.log -Tail 5
Select-String -Path D:\Docker\mounted-configs\sam2-trainer\<run>\train.log -Pattern 'saved best_model|early stopping at'
```

- `saved best_model.pt` marks the best epoch.
- `early stopping at epoch N` or `Training complete` means the run has finished.
- `/outputs` in the container is `D:\Docker\mounted-configs\sam2-trainer` on the host (`-sem` for SEM).
- Runs and checkpoints are at `/storage4/Sam2Trainer/runs/<run>`.

Update `docs/status.md` when what you find has changed, and give the date.

## 2. Where things are

| What | Where |
|---|---|
| Trainer container | `sam2-trainer-cursor-dev-1`. Start it with `docker\run-cursor-dev.ps1 -Modality TEM\|SEM` (README). |
| Galleries | `sam2-trainer-annotation-gallery-{tem,sem}-1` behind `gallery-router` |
| Segmentation server | `viking-segmentation-latest-1` (code in the Viking repo). It shares the GPU with training. |
| Correction service | `viking-section-correction-service-1`. Its sets are mounted at `/corrections`. |
| Training data | Local mirror at `/data-local/<set>`; the NAS is at `/storage4`. Sync scripts: `scripts/WSL`, `scripts/NAS`. |
| Config | `sam2_segmentation_trainer/config/em_finetune.yaml` (Hydra; override with `key=value`) |
| Prompt encoding (shared with the server) | `annotation_crops/prompt_encoding.py`, `PROMPT_ENCODING_VERSION` |
| Frozen validation splits and gold sets | `/storage4/Sam2Trainer/valsets`, `sam2-em-valsets` (`docs/validation_sets.md`) |
| Training-data exporter | The nornir repo: `.cursor/skills/segmentation-training-export/SKILL.md` |
| Run metrics (REST) | The `sam2-training-report` and `tensorboard-api` skills |

## 3. Which doc answers what

The specs and the code are the source of truth. Session summaries are not.

- Goals, prompt mix, the signed mask convention: `docs/overall_training_plan.md`
- Edge confidence, prompt families, loss, validation, production requirements, open questions (§11):
  `docs/mask_refinement_plan.md`
- Prompt families with pictures: `docs/prompt_overview.md`
- Validation sets and gold sets: `docs/validation_sets.md`
- Carried circles and polygons from adjacent sections: `docs/adjacent_circle_seed_plan.md`
- v7 seed trust: `docs/depth_relative_trust.md`; the seed-taper measurement: `docs/circle_seed_trust_eval.md`
- Model and set naming: `naming-conventions.mdc`; model sidecars: `docs/model_manifest.md`
- Citations, formulas, decisions and their reasons: `docs/publication_notes.md` (the `publication-notes` skill).
  Add an entry there whenever a decision is made.

## 4. Touching a run

- **Ask before pausing, stopping or restarting** a run or a server you did not start, including the segmentation
  servers. Don't restart `viking-segmentation-v4-1`.
- **To pause,** send SIGINT to the trainer. It finishes the step, writes `last.pt` and exits.
- **To resume,** use `sam2-em-train training.run_name=<run>` with the same `data.*` overrides it was started
  with (v5: `data.root=/data-local/v3 data.registration_root=/storage4 data.corrections_root=/tmp/corrections`). Write a
  launcher script only in your scratchpad or `/outputs/_scratch`, and delete it once the run is back. Resume refuses a different prompt layout, `freeze_mode`,
  `unfreeze_mask_cnn` or optimizer groups. A batch-size change resumes at the same sample.
- **To start a new run,** use `training.run_name=auto` with `-refresh` from the pretrained weights, not a
  resume of an older run. Pass `data.frozen_split=v5` so its validation scores stay comparable.
- **To evaluate offline,** use `python -m sam2_segmentation_trainer.evaluate --goal --checkpoint <serve .pt>
  --split v5 --root /data-local/<set> --set eval.val_prompts=[...]`. An evaluation needs the GPU, so pause
  training first (ask).
- **Model selection** uses a 10% validation subset. Noise floors: core recall 0.004, IoU 0.006, spill 0.006,
  merge rate 0.002, fragmentation 0.012. Patience is 3.
- **GPU sharing.** Training, the segmentation server and the Viking client share one 23 GB GPU. When it is
  oversubscribed, a training step slows from about 0.3 s to 20 s or more, and tile embedding slows from 0.5 s
  to about 30 s. Check `nvidia-smi` before blaming code.

## 5. Working in the repo

- Read `CLAUDE.md`. The rules are in `.cursor/rules`. After editing one, run `python .claude/sync_claude_rules.py`.
- Other sessions edit the same checkout. Stage by path, and never commit someone else's uncommitted changes.
  `git status` before committing.
- A change that the server, the exporter or the Viking client reads also needs the `cross-repo-change` skill.
