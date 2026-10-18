"""The model manifest: ``<weights stem>.model.json``, the sidecar that travels with a served model.

It tells the segmentation server and its clients what the model is (id, version, the training data it saw), how it
must be fed (tiled or whole screen) and what each side does about the mask prompt (the encoding, and what the client
must still do before it sends a mask). The schema is documented in ``docs/model_manifest.md``; the segmentation
server has its own reader for it and refuses a model whose manifest it cannot honour.

The trainer writes one for every export (``export_serve.py``). Models that predate the exporter (v2, v3) get a
hand-written one: ``python -m sam2_segmentation_trainer.model_manifest check <file>`` rejects a misspelled or
missing field. ``unknown`` is a legal value for fields that cannot be reconstructed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
CONTRACT_VERSION = 1
MANIFEST_SUFFIX = ".model.json"

SUBMISSION_MODES = ("tile_grid", "full_viewport")
RESOLUTION_MODES = ("single", "multi")
LEGACY_ENCODING = "legacy-flat-v1"
"""The mask prompt the server built before the trainer's encoding existed: a flat +8 inside a binary mask, -8 for a
background mask, -2 for a background rectangle."""

TOP_LEVEL_KEYS = (
    "schema_version",
    "model",
    "training_data",
    "code",
    "capabilities",
    "prompt_encoding",
    "client_contract",
    "server_contract",
)
REQUIRED_KEYS = ("schema_version", "model", "capabilities", "prompt_encoding", "client_contract", "server_contract")
MODEL_KEYS = (
    "model_id", "model_version", "display_name", "description", "exported_at", "run", "epoch", "base_checkpoint"
)
MODEL_ID_RE = re.compile(r"^v\d+$")
MODEL_VERSION_RE = re.compile(r"^v\d+\.\d+$")
MODEL_STEM_RE = re.compile(r"^(?:[A-Za-z0-9_]+-)*(v\d+\.\d+)$")
"""A weights file stem: ``v<VERSION>.<EPOCH>``, optionally led by labels joined with ``-`` (``sem-v1.5``). The labels
are for people sorting files; the manifest's ``model_id`` and ``model_version`` never carry them."""
_RUN_NUMBER_RE = re.compile(r"_v(\d+)$")
CLIENT_CONTRACT_KEYS = (
    "contract_version",
    "mask_format",
    "window",
    "circle_diameter_scale",
    "mask_roles",
    "background_mask",
    "requires_model_id",
    "notes",
)
SERVER_CONTRACT_KEYS = ("builds_mask_input", "encoding")
CAPABILITY_KEYS = ("submission_mode", "resolution_mode")
MASK_ROLES = ("seed", "correction")


def model_id_for_run(run_name: str) -> str | None:
    """``v<VERSION>`` for a run named ``<prefix>_v<VERSION>`` (``sam2_em_v4`` is ``v4``); None for any other name."""
    match = _RUN_NUMBER_RE.search(run_name)
    return f"v{int(match.group(1))}" if match else None


def model_version_name(model_id: str, epoch: int) -> str:
    """``v<VERSION>.<EPOCH>``: the model id and the trainer's epoch index (the one in the log, from 0)."""
    return f"{model_id}.{int(epoch)}"


def stem_model_version(stem: str) -> str | None:
    """The ``v<VERSION>.<EPOCH>`` a weights stem ends with (``sem-v1.5`` gives ``v1.5``); None for any other stem."""
    match = MODEL_STEM_RE.match(stem)
    return match.group(1) if match else None


def stem_mismatch(stem: str, model_version: str) -> str | None:
    """A description of the problem when ``stem`` names a model version other than ``model_version``, else None.

    A stem that does not follow the convention at all (``serve_epoch5``, ``sam2_em_v3_epoch_9``) is not a mismatch:
    existing names stay as they are.
    """
    named = stem_model_version(stem)
    if named is None or named == model_version:
        return None
    return f"the file name {stem!r} says {named} but the model version is {model_version}"


def manifest_path(weights: Path) -> Path:
    """``<weights stem>.model.json`` next to ``weights``."""
    weights = Path(weights)
    return weights.with_name(f"{weights.stem}{MANIFEST_SUFFIX}")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_manifest(
    *,
    model_id: str,
    model_version: str,
    encoding: Mapping[str, Any],
    training_data: Mapping[str, Any],
    run: str | None = None,
    epoch: int | None = None,
    display_name: str | None = None,
    description: str = "",
    base_checkpoint: str | None = None,
    submission_mode: str = "tile_grid",
    resolution_mode: str = "single",
) -> dict[str, Any]:
    """The manifest of a model the trainer exported with ``encoding`` (the dict of ``encoding_metadata``).

    ``model_id`` is ``v<VERSION>`` and ``model_version`` is ``v<VERSION>.<EPOCH>`` (see
    ``.cursor/rules/naming-conventions.mdc``). ``description`` is for the operator to fill in; an export starts it empty.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "model_id": model_id,
            "model_version": model_version,
            "display_name": display_name or model_id,
            "description": description,
            "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "run": run,
            "epoch": epoch,
            "base_checkpoint": base_checkpoint,
        },
        "training_data": dict(training_data),
        "code": {"trainer_git_sha": _git_sha(), "prompt_encoding": encoding.get("prompt_encoding")},
        "capabilities": {"submission_mode": submission_mode, "resolution_mode": resolution_mode},
        "prompt_encoding": dict(encoding),
        "client_contract": {
            "contract_version": CONTRACT_VERSION,
            "mask_format": "png-l8-binary",
            "window": 1024,
            "circle_diameter_scale": float(encoding.get("circle_diameter_scale", 1.0)),
            "mask_roles": list(MASK_ROLES),
            "background_mask": "neighbour-annotations-union",
            "requires_model_id": True,
            "notes": (
                "The client sends binary masks and labels the foreground mask seed or correction. It scales a circle "
                "seed's diameter by circle_diameter_scale. The server builds mask_input. Neighbouring annotations "
                "arrive as one merged background mask, so each neighbour's own edge distance and a circle "
                "annotation's radius trust are approximated."
            ),
        },
        "server_contract": {"builds_mask_input": True, "encoding": encoding.get("prompt_encoding")},
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``manifest`` as JSON, creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _unknown(label: str, given: Mapping[str, Any], allowed: tuple[str, ...]) -> list[str]:
    return [f"{label}: unknown field {key!r} (allowed: {', '.join(allowed)})" for key in given if key not in allowed]


def validate_manifest(manifest: object) -> list[str]:
    """Problems with ``manifest`` (empty when it is acceptable): a misspelled or missing field, a bad enum value."""
    if not isinstance(manifest, dict):
        return ["the manifest must be a JSON object"]
    problems = _unknown("manifest", manifest, TOP_LEVEL_KEYS)
    problems += [f"manifest: missing {key!r}" for key in REQUIRED_KEYS if key not in manifest]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}")

    sections: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("model", MODEL_KEYS),
        ("capabilities", CAPABILITY_KEYS),
        ("client_contract", CLIENT_CONTRACT_KEYS),
        ("server_contract", SERVER_CONTRACT_KEYS),
    )
    for name, allowed in sections:
        section = manifest.get(name)
        if section is None:
            continue
        if not isinstance(section, dict):
            problems.append(f"{name} must be an object")
            continue
        problems += _unknown(name, section, allowed)

    model = manifest.get("model")
    if isinstance(model, dict):
        for key in ("model_id", "model_version"):
            if not isinstance(model.get(key), str) or not model.get(key):
                problems.append(f"model.{key} must be a non-empty string")
        if isinstance(model.get("model_id"), str) and not MODEL_ID_RE.match(model["model_id"]):
            problems.append(f"model.model_id {model['model_id']!r} must look like v4 (v<VERSION>)")
        if isinstance(model.get("model_version"), str) and not MODEL_VERSION_RE.match(model["model_version"]):
            problems.append(f"model.model_version {model['model_version']!r} must look like v4.3 (v<VERSION>.<EPOCH>)")
        if "description" in model and not isinstance(model["description"], str):
            problems.append("model.description must be a string")
    caps = manifest.get("capabilities")
    if isinstance(caps, dict):
        if caps.get("submission_mode") not in SUBMISSION_MODES:
            problems.append(f"capabilities.submission_mode must be one of {SUBMISSION_MODES}")
        if caps.get("resolution_mode") not in RESOLUTION_MODES:
            problems.append(f"capabilities.resolution_mode must be one of {RESOLUTION_MODES}")
    server = manifest.get("server_contract")
    if isinstance(server, dict) and not isinstance(server.get("encoding"), str):
        problems.append("server_contract.encoding must be a string (use 'legacy-flat-v1' for the old encoding)")
    client = manifest.get("client_contract")
    if isinstance(client, dict):
        roles = client.get("mask_roles", [])
        if not isinstance(roles, list) or any(role not in MASK_ROLES for role in roles):
            problems.append(f"client_contract.mask_roles must be a list drawn from {MASK_ROLES}")
        if "requires_model_id" in client and not isinstance(client["requires_model_id"], bool):
            problems.append("client_contract.requires_model_id must be true or false")
    encoding = manifest.get("prompt_encoding")
    if encoding is not None and not isinstance(encoding, dict):
        problems.append("prompt_encoding must be an object")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a model manifest (<weights stem>.model.json)")
    parser.add_argument("command", choices=["check"])
    parser.add_argument("file", type=Path, nargs="+")
    args = parser.parse_args(argv)
    bad = 0
    for path in args.file:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            problems = validate_manifest(manifest)
            version = manifest.get("model", {}).get("model_version") if isinstance(manifest, dict) else None
            if isinstance(version, str):
                mismatch = stem_mismatch(path.name.removesuffix(MANIFEST_SUFFIX), version)
                if mismatch:
                    problems.append(mismatch)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            problems = [f"cannot read: {exc}"]
        print(f"{path}: {'ok' if not problems else 'INVALID'}")
        for problem in problems:
            print(f"  {problem}")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
