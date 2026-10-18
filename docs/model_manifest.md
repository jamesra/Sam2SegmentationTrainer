# Model manifest (`<weights stem>.model.json`)

The sidecar that travels with a served model. The segmentation server reads it at start-up and reports it in
`GetServerStatus`; clients read it from there. It lives next to the weights, with the same stem:

```
D:\Docker\Run\segmentation-server\
  v4.3.pt
  v4.3.model.json                     <- this file
  v4.3.prompt_encoding.json           (older record, still written)
```

`sam2-em-export-serve` (and the per-epoch export that training starts) writes it for every export. Models that predate
the exporter (v2, v3) need a hand-written one: start from `docs/model_manifest_examples/` and check it with

```bash
python -m sam2_segmentation_trainer.model_manifest check path/to/model.model.json
```

The check rejects a misspelled or missing field and a bad enum value. `"unknown"` is a legal value wherever a fact cannot
be reconstructed. A model file with no manifest is treated by the server as a tiled, legacy-encoding model, with a
warning.

## Fields (schema_version 1)

| Section | Field | Meaning |
|---------|-------|---------|
| (top) | `schema_version` | `1` |
| `model` | `model_id` | The model's version number, `v<VERSION>`: `v4`. Clients send it with requests; the server refuses a mismatch. Not the training data's number. |
| | `model_version` | One export of that model, `v<VERSION>.<EPOCH>`: `v4.3`, the trainer's epoch index counted from 0. |
| | `description` | What the model is: what changed, what to use it for. **The operator writes it.** The exporter starts it empty and keeps what is there when it re-exports. The server reports it to clients. |
| | `display_name`, `exported_at`, `run`, `epoch`, `base_checkpoint` | Informational. |
| `training_data` | `set`, `requested`, `fingerprint`, `volumes`, `train_examples`, `val_examples`, `split_seed` | Copied from the run's `training_data.json` (`data_record.py`). **The data version, kept apart from the model version.** `set` is the data folder (`v3`), `fingerprint` hashes the manifest, approved and ignore files. |
| `code` | `trainer_git_sha`, `prompt_encoding` | What built the export. |
| `capabilities` | `submission_mode` | `tile_grid` (fixed 1024 px cells, `SegmentTilesStream`) or `full_viewport` (the whole screen, `UploadImage` + `SegmentImage`). |
| | `resolution_mode` | `single` or `multi` (the client may submit a coarser downsample). Not consulted for `full_viewport`. |
| `prompt_encoding` | `prompt_encoding`, `EDGE_TAU_PX`, ... | The constants the mask prompt was trained with (`annotation_crops/prompt_encoding.py` `encoding_metadata`). Legacy models: `{"prompt_encoding": "legacy-flat-v1"}`. |
| `client_contract` | `contract_version` | Bump when the client's duties change. |
| | `mask_format` | `png-l8-binary`: 1-bit or 8-bit greyscale PNG, non-zero is in the mask. |
| | `window` | Prompt window in pixels (1024). |
| | `circle_diameter_scale` | The client scales a circle seed's diameter by this before sending. |
| | `mask_roles` | Roles the client must label the foreground mask with: `seed`, `correction`. Empty for legacy models. |
| | `background_mask` | What the background mask holds (`neighbour-annotations-union`: one merged mask). |
| | `requires_model_id` | If true, a request without `model_id` is rejected. |
| `server_contract` | `builds_mask_input` | The server turns the masks into the 256x256 `mask_input`. |
| | `encoding` | The encoding it uses. The server refuses to start if it does not implement it. |

## Encodings the server implements

- `legacy-flat-v1`: +8 inside a binary foreground mask, -8 for a background mask, -2 for a background rectangle.
- `edge-confidence-v6`: `annotation_crops/prompt_encoding.py` with the manifest's constants.

## Hand-written examples

- `model_manifest_examples/sam2_em_v3.model.json`: tiled, legacy encoding, no roles, no model id required.
- `model_manifest_examples/sam2_em_v2.model.json`: whole screen, points only.

Fill in `description`, `training_data.set` and the file name (`v3.9.pt` with `v3.9.model.json`, or `sem-v3.9.pt` with `sem-v3.9.model.json`: the stem may lead with labels joined by `-`) to match your files before
copying them next to the weights. The check also rejects a `model_id` that is not `v<VERSION>` or a `model_version` that
is not `v<VERSION>.<EPOCH>` (neither takes a label), and a file name that ends in a different version than `model_version`.
