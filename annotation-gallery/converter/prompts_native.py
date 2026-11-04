"""ctypes binding of Viking.SegmentationPrompts.Native, the shared builder of location segmentation requests.

The library is Viking's (``Viking\\SegmentationPrompts``); the gallery image builds it for Linux in its first stage and
puts it at ``/app/native`` (``GALLERY_PROMPTS_LIBRARY`` overrides the path). Calls take and return serialized protobuf
messages (``seg_proto/prompt_input.proto``), so the Viking client and the gallery send the same bytes for the same
location. See the library's README for the rules it applies.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Optional, Sequence, Tuple

from seg_proto import prompt_input_pb2 as prompt_pb
from seg_proto import segmentation_pb2 as pb

DEFAULT_LIBRARY = "/app/native/Viking.SegmentationPrompts.Native.so"

_OK, _INVALID_ARGUMENT, _TOO_LARGE = 0, 1, 2
_BYTES = ctypes.POINTER(ctypes.c_uint8)


class PromptRejected(ValueError):
    """The library refused the input (no targets, an unsupported type, mixed circles and areas, bad geometry)."""


class PromptTooLarge(ValueError):
    """The targets cover more tiles or shapes than one request may carry."""


class PromptLibrary:
    """One loaded copy of the native library."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.environ.get("GALLERY_PROMPTS_LIBRARY") or DEFAULT_LIBRARY
        library = ctypes.CDLL(self.path)
        out = [ctypes.POINTER(_BYTES), ctypes.POINTER(ctypes.c_ssize_t)]
        library.sp_build_start.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, *out]
        library.sp_reach.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, *out]
        library.sp_background_for.argtypes = [
            ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_char_p, ctypes.c_ssize_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_ssize_t, *out,
        ]
        for name in ("sp_build_start", "sp_reach", "sp_background_for"):
            getattr(library, name).restype = ctypes.c_int
        library.sp_free.argtypes = [_BYTES]
        library.sp_free.restype = None
        self._library = library

    def _call(self, name: str, *args) -> bytes:
        buffer, length = _BYTES(), ctypes.c_ssize_t()
        status = getattr(self._library, name)(*args, ctypes.byref(buffer), ctypes.byref(length))
        try:
            payload = ctypes.string_at(buffer, length.value)
        finally:
            self._library.sp_free(buffer)
        if status == _OK:
            return payload
        message = payload.decode("utf-8", errors="replace")
        if status == _INVALID_ARGUMENT:
            raise PromptRejected(message)
        if status == _TOO_LARGE:
            raise PromptTooLarge(message)
        raise RuntimeError(f"{name} failed: {message}")

    def build_start(self, prompt_input: "prompt_pb.PromptInput") -> "pb.SegmentTilesRequest":
        data = prompt_input.SerializeToString()
        return pb.SegmentTilesRequest.FromString(self._call("sp_build_start", data, len(data)))

    def reach(self, prompt_input: "prompt_pb.PromptInput") -> Tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)``: every annotation meeting it must be passed as a candidate."""
        data = prompt_input.SerializeToString()
        rect = prompt_pb.PromptReach.FromString(self._call("sp_reach", data, len(data))).reach
        return rect.x_min, rect.y_min, rect.x_max, rect.y_max

    def background_for(
        self, prompt_input: "prompt_pb.PromptInput", needed: "pb.TilesNeeded", sent_ids: Sequence[int]
    ) -> "prompt_pb.PromptBackground":
        data = prompt_input.SerializeToString()
        asked = needed.SerializeToString()
        sent = (ctypes.c_int64 * max(1, len(sent_ids)))(*sent_ids)
        return prompt_pb.PromptBackground.FromString(
            self._call("sp_background_for", data, len(data), asked, len(asked), sent, len(sent_ids))
        )


_LOCK = threading.Lock()
_SHARED: Optional[PromptLibrary] = None


def library() -> PromptLibrary:
    """The process's library, loaded on first use."""
    global _SHARED
    with _LOCK:
        if _SHARED is None:
            _SHARED = PromptLibrary()
        return _SHARED
