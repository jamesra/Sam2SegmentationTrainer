---
name: sam2-grounding
description: Grounds SAM2 (Meta Segment Anything 2) plans in pinned-commit theory and a complete API map. Use when planning or editing SAM2 fine-tuning, export-for-serve, checkpoints, Hydra configs, or Meta sam2 API usage in this trainer. Canonical procedure lives in the Viking repo.
---

# SAM2 grounding (trainer pointer)

**Canonical skill:** sibling checkout  
`../Viking/.cursor/skills/sam2-grounding/SKILL.md`  
(absolute: `d:/src/git/Viking/.cursor/skills/sam2-grounding/SKILL.md`)

Also read that folder’s `reference.md` for theory sources and the API-map checklist.

## This repo’s obligations

1. Open the Viking skill and follow its grounding checklist before SAM2-related plans or code.
2. Resolve the pin from this repo’s `SAM2_GIT_SHA` (docker/env) and confirm it matches  
   `Viking/Servers/SegmentationServer/Dockerfile` `ARG SAM2_GIT_SHA`, or note deliberate drift to James.
3. Pin notify rule: [sam2-pin.mdc](../../rules/sam2-pin.mdc). Do not silently unpin.
4. Serve weights must be Meta `build_sam2` format (`{"model": state_dict}`); use `sam2-em-export-serve` for SegmentationServer mounts.

Do not duplicate the theory/API procedure here — keep one source of truth in Viking.
