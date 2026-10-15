"""The prompt-encoding record: which ``mask_input`` encoding a model was trained with, kept next to the weights.

The record is the dict of :func:`annotation_crops.prompt_encoding.encoding_metadata`. It lives in three places so
no copy of the weights loses it:

* ``prompt_encoding.json`` in the run directory, written when training starts (a raw ``best_model.pt`` has it);
* ``<weights stem>.prompt_encoding.json`` next to an exported model, and the same dict inside the exported ``.pt``
  under ``"prompt_encoding"``.

Evaluation and validation read it back so a model is scored with the encoding it was trained on; a value given on
the command line overrides it and the override is logged. The exporter never scales circles; the trainer applies
every scale and trust fraction recorded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from annotation_crops import prompt_encoding as enc
from annotation_crops.negatives import DEFAULT_CIRCLE_DIAMETER_SCALE
from annotation_crops.prompt_encoding import CircleAnnotationTrust

RECORD_NAME = "prompt_encoding.json"
"""File name of the record in a run directory."""

_SCALAR_KEYS = {
    "edge_tau_px": "EDGE_TAU_PX",
    "edge_tau_depth_frac": "EDGE_TAU_DEPTH_FRAC",
    "seed_logit": "SEED_LOGIT",
    "correction_logit": "CORRECTION_LOGIT",
    "circle_diameter_scale": "circle_diameter_scale",
}
"""``prompt.*`` config key -> key in the record."""


def sidecar_path(weights: Path) -> Path:
    """``<weights stem>.prompt_encoding.json`` next to ``weights``."""
    weights = Path(weights)
    return weights.with_name(f"{weights.stem}.prompt_encoding.json")


def write_record(path: Path, record: Mapping[str, Any]) -> Path:
    """Write ``record`` as JSON, creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def settings_from_cfg(cfg: Any) -> dict[str, Any]:
    """The encoding values of a training config (``prompt.*``); a key it lacks keeps the library default.

    Keys: ``edge_tau_px``, ``edge_tau_depth_frac``, ``seed_logit``, ``correction_logit``, ``circle_diameter_scale``
    (floats) and ``circle_annotation_trust`` (:class:`CircleAnnotationTrust`, validated).
    """
    from omegaconf import OmegaConf

    defaults = {
        "edge_tau_px": enc.EDGE_TAU_PX,
        "edge_tau_depth_frac": enc.EDGE_TAU_DEPTH_FRAC,
        "seed_logit": enc.SEED_LOGIT,
        "correction_logit": enc.CORRECTION_LOGIT,
        "circle_diameter_scale": DEFAULT_CIRCLE_DIAMETER_SCALE,
    }
    settings: dict[str, Any] = {
        key: float(OmegaConf.select(cfg, f"prompt.{key}", default=default)) for key, default in defaults.items()
    }
    raw = OmegaConf.select(cfg, "prompt.circle_annotation_trust", default=None)
    if raw is not None and not isinstance(raw, Mapping):
        raw = OmegaConf.to_container(raw)
    settings["circle_annotation_trust"] = CircleAnnotationTrust.from_mapping(raw)
    return settings


def metadata_from_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The record for ``settings`` (the dict of :func:`settings_from_cfg`; missing keys keep the defaults)."""
    rest = dict(settings)
    scale = float(rest.pop("circle_diameter_scale", DEFAULT_CIRCLE_DIAMETER_SCALE))
    return enc.encoding_metadata(scale, **rest)


def read_record(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    """The record for a model, and where it came from (``""`` when there is none).

    ``path`` is a weights file or a run directory. A directory is read for ``prompt_encoding.json``. A file is read
    for its sidecar, then for ``prompt_encoding.json`` in its folder (the run), then for the ``"prompt_encoding"``
    entry of an exported ``.pt``.
    """
    if path is None:
        return None, ""
    path = Path(path)
    candidates = [path / RECORD_NAME] if path.is_dir() else [sidecar_path(path), path.with_name(RECORD_NAME)]
    for candidate in candidates:
        if candidate.is_file():
            try:
                record = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                print(f"WARNING: could not read {candidate}: {exc}")
                continue
            if isinstance(record, dict):
                return record, str(candidate)
    if path.is_file() and path.suffix == ".pt":
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # a file that is not a checkpoint has no record
            print(f"WARNING: could not read {path} for its prompt encoding: {exc}")
            return None, ""
        embedded = payload.get("prompt_encoding") if isinstance(payload, dict) else None
        if isinstance(embedded, dict):
            return embedded, f"{path} (embedded)"
    return None, ""


@dataclass(slots=True)
class ResolvedEncoding:
    """The encoding an evaluation uses: the defaults, then the model's record, then command-line overrides.

    ``explicit`` holds the values that came from the record or the command line (not the defaults), by ``prompt.*``
    config key, so a config-driven run is only changed where the model or the user said so.
    """

    circle_diameter_scale: float = DEFAULT_CIRCLE_DIAMETER_SCALE
    edge_tau_px: float = enc.EDGE_TAU_PX
    edge_tau_depth_frac: float = enc.EDGE_TAU_DEPTH_FRAC
    seed_logit: float = enc.SEED_LOGIT
    correction_logit: float = enc.CORRECTION_LOGIT
    circle_annotation_trust: CircleAnnotationTrust = enc.DEFAULT_CIRCLE_TRUST
    explicit: dict[str, Any] = field(default_factory=dict)

    def prompt_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :class:`sam2_segmentation_trainer.data.prompts.PromptConfig`."""
        return {
            "edge_tau_px": self.edge_tau_px,
            "edge_tau_depth_frac": self.edge_tau_depth_frac,
            "seed_logit": self.seed_logit,
            "correction_logit": self.correction_logit,
            "circle_annotation_trust": self.circle_annotation_trust,
        }

    def cfg_overrides(self) -> list[str]:
        """``key=value`` config overrides for the values that are explicit (record or command line)."""
        out: list[str] = []
        for key, value in self.explicit.items():
            out.append(f"prompt.{key}={value}")
        return out


def resolve_encoding(
    record: Mapping[str, Any] | None,
    *,
    source: str = "",
    circle_scale: float | None = None,
    full_frac: float | None = None,
    zero_frac: float | None = None,
    log: Callable[[str], None] = print,
) -> ResolvedEncoding:
    """Defaults, overlaid with the model's ``record`` when there is one, overlaid with command-line overrides.

    Each override is logged, with the value it replaced. A bad trust pair (``full_frac >= zero_frac``) raises.
    """
    resolved = ResolvedEncoding()
    if record:
        for key, record_key in _SCALAR_KEYS.items():
            if record_key in record:
                setattr(resolved, key, float(record[record_key]))
                resolved.explicit[key] = float(record[record_key])
        trust = record.get("circle_annotation_trust")
        if isinstance(trust, Mapping):
            resolved.circle_annotation_trust = CircleAnnotationTrust.from_mapping(trust)
            resolved.explicit["circle_annotation_trust.full_frac"] = resolved.circle_annotation_trust.full_frac
            resolved.explicit["circle_annotation_trust.zero_frac"] = resolved.circle_annotation_trust.zero_frac
        version = record.get("prompt_encoding")
        log(f"prompt encoding from {source or 'model record'}: {version}, trust {resolved.circle_annotation_trust.as_dict()}")
        if version != enc.PROMPT_ENCODING_VERSION:
            log(f"WARNING: the model's encoding is {version}, this code builds {enc.PROMPT_ENCODING_VERSION}")
    else:
        log("no prompt-encoding record for this model; using the library defaults")

    if circle_scale is not None:
        log(f"override: circle_diameter_scale {resolved.circle_diameter_scale} -> {circle_scale}")
        resolved.circle_diameter_scale = float(circle_scale)
        resolved.explicit["circle_diameter_scale"] = float(circle_scale)
    if full_frac is not None or zero_frac is not None:
        old = resolved.circle_annotation_trust
        resolved.circle_annotation_trust = CircleAnnotationTrust(
            old.full_frac if full_frac is None else float(full_frac),
            old.zero_frac if zero_frac is None else float(zero_frac),
        )
        new = resolved.circle_annotation_trust
        log(
            f"override: circle_annotation_trust ({old.full_frac}, {old.zero_frac}) -> ({new.full_frac}, {new.zero_frac})"
        )
        resolved.explicit["circle_annotation_trust.full_frac"] = new.full_frac
        resolved.explicit["circle_annotation_trust.zero_frac"] = new.zero_frac
    return resolved
