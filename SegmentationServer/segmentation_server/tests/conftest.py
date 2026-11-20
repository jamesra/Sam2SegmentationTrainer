"""Make sibling packages importable when pytest is launched from this project."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SEGMENTATION_SERVER_ROOT = _PROJECT_ROOT
_SEGMENTATION_GRPC_ROOT = _PROJECT_ROOT.parent / "segmentation_grpc"
# The Sam2Trainer repository root, for the shared prompt_encoders package.
_REPOSITORY_ROOT = _PROJECT_ROOT.parents[1]

for path in (_SEGMENTATION_SERVER_ROOT, _SEGMENTATION_GRPC_ROOT, _REPOSITORY_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
