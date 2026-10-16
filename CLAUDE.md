# Sam2Trainer – instructions for Claude Code

The rules and skills for this repo are authored for Cursor and live in `.cursor/`. They are the single source of
truth; this file links to them instead of copying them, so edits in `.cursor/` apply to both tools.
User-level rules are imported from `~/.claude/CLAUDE.md` (which links `~/.cursor/rules`).

## Path-scoped rules (read the file before editing anything matching its globs)

The `.mdc` frontmatter `globs:` line says which paths each rule covers.

| Rule | Applies to |
|------|-----------|
| `.cursor/rules/sam2-pin.mdc` | `docker/Dockerfile` and SAM2 integration (pin `SAM2_GIT_SHA`; tell James when upstream `main` is newer, never unpin silently) |
| `.cursor/rules/naming-conventions.mdc` | Naming or exporting training sets (`v#.#`) and models (`v<VERSION>.<EPOCH>`), their metadata sidecars (`training_set.json`, `<stem>.model.json` with an operator `description`), `scripts/WSL/**`, `export_serve.py`, `model_manifest.py`, `data_record.py`. Sets and models have separate version numbers; say which one you mean. |

## Skills (`.cursor/skills/<name>/SKILL.md`)

Same `SKILL.md` format as Claude skills. Read the SKILL.md when the task matches. If `.claude/skills` exists
(a local junction to `.cursor/skills`), they also appear in the Skill tool.

| Skill | Use for |
|-------|---------|
| `sam2-grounding` | Planning or editing SAM2 fine-tuning, export-for-serve, checkpoints, Hydra configs or Meta `sam2` API use. Points at the canonical skill in the Viking repo. |