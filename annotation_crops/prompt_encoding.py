"""Moved to :mod:`prompt_encoders.v6` on 2026-11-20 (shared by the trainer and the segmentation server).

This name stays so the gallery, the converter and older imports keep working: importing it yields the very same module
object as ``prompt_encoders.v6``, so constants, functions and monkeypatches are shared.
"""

import sys

import prompt_encoders.v6 as _moved

sys.modules[__name__] = _moved
