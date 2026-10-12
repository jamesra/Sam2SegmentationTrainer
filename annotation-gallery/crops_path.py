"""Put the trainer's ``annotation_crops`` package on ``sys.path`` when it sits beside this folder."""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_repo = _here.parent
if (_repo / "annotation_crops" / "__init__.py").is_file():
    root = str(_repo)
    if root not in sys.path:
        sys.path.insert(0, root)
