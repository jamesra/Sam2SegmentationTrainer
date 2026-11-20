# Sam2Trainer – instructions for Claude Code

This repository holds the SAM2 trainer (`sam2_segmentation_trainer/`, `annotation_crops/`), the annotation gallery
(`annotation-gallery/`) and, since 2026-11-20, the Viking segmentation server (`SegmentationServer/`, moved from Viking's
`Servers/SegmentationServer` with its history; its `README.md` describes it). `gRPC_Protos/` is a submodule
(jamesra/viking_gRPC_protos): run `git submodule update --init` after cloning. A prompt-encoding change lands on the
trainer and the server in one commit (`serve-train-prompt-parity.mdc`). The server deploys as stack
`viking-segmentation` from `deploy/stacks.psd1` (connectomes-deploy, `SAM2TRAINER_REPO_ROOT_HOST`).

The rules and skills for this repo are authored for Cursor and live in `.cursor/`. They are the single source of
truth; this file links to them instead of copying them, so edits in `.cursor/` apply to both tools.
User-level rules are imported from `~/.claude/CLAUDE.md` (which links `~/.cursor/rules`).

## Path-scoped rules (loaded automatically from `.claude/rules/`)

Each `.cursor/rules/*.mdc` rule with `globs:` is mirrored into `.claude/rules/<name>.md` with a `paths:` list, so Claude
loads it when it reads or edits a matching file. After adding or editing a `.cursor/rules` `.mdc` file, rerun
`python .claude/sync_claude_rules.py` and commit the result; never edit `.claude/rules/` by hand
(`python .claude/sync_claude_rules.py --check` reports drift). The table is an index.

| Rule | Applies to |
|------|-----------|
| `.cursor/rules/sam2-pin.mdc` | `docker/Dockerfile`, `SegmentationServer/Dockerfile` and SAM2 integration (one `SAM2_GIT_SHA` for trainer and server; tell James when upstream `main` is newer, never unpin silently) |
| `.cursor/rules/circle-centre-xy.mdc` | Circle annotations in the converter, gallery and exporter: centre is OData `X`, `Y`, never the first point or any average of the `CIRCULARSTRING` ring; `Radius` is mosaic-only. |
| `.cursor/rules/naming-conventions.mdc` | Naming or exporting training sets (`v#.#`) and models (`v<VERSION>.<EPOCH>`), their metadata sidecars (`training_set.json`, `<stem>.model.json` with an operator `description`), `scripts/WSL/**`, `export_serve.py`, `model_manifest.py`, `data_record.py`. Sets and models have separate version numbers; say which one you mean. |
| `.cursor/rules/serve-train-prompt-parity.mdc` | `prompt_encoding.py`, prompt building, export, plan docs: serving must encode prompts exactly as training does; bump `PROMPT_ENCODING_VERSION` on any change; lists the five known serve-time mismatches in the server's tiled walk. |
| `.cursor/rules/circle-prompts.mdc` | Circle prompts in data, config and plan docs: the trainer scales radii, the exporter never does; annotator circles are negatives only; the seed taper is trained, not served on v5; pointers to the v7 trust and carried-circle plans. |

## Skills (`.cursor/skills/<name>/SKILL.md`)

Same `SKILL.md` format as Claude skills. Read the SKILL.md when the task matches. If `.claude/skills` exists
(a local junction to `.cursor/skills`), they also appear in the Skill tool.

| Skill | Use for |
|-------|---------|
| `sam2-grounding` | Planning or editing SAM2 fine-tuning, export-for-serve, the segmentation server's inference, checkpoints, Hydra configs or Meta `sam2` API use (canonical copy, moved from Viking 2026-11-20). |
| `sam2trainer-startup` | Start of a session, or before pausing, resuming, evaluating or exporting a run: check `docs/status.md` against the logs, where everything is, which doc answers what, safe run handling. |
