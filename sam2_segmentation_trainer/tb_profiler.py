"""Bounded ``torch.profiler`` traces for the TensorBoard PYTORCH_PROFILER tab.

A full-run trace of SAM2 is large enough to stall the exporting step and fill
the checkpoint share. The schedule records a few batches near process start,
writes ``*.pt.trace.json.gz`` under the run's TensorBoard directory, then
``step`` stays idle. ``training.profiler.enabled=false`` turns it off.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

_DEFAULTS = {
    "enabled": True,
    "skip_first": 5,
    "wait": 1,
    "warmup": 1,
    "active": 3,
    "repeat": 1,
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": False,
    "with_modules": True,
}


def profiler_options(cfg: DictConfig) -> dict:
    """Resolved profiler settings. Missing keys use :data:`_DEFAULTS`."""
    raw = OmegaConf.select(cfg, "training.profiler")
    node = {} if raw is None else OmegaConf.to_container(raw, resolve=True)
    if not isinstance(node, dict):
        node = {}
    out = dict(_DEFAULTS)
    out.update({key: node[key] for key in _DEFAULTS if key in node})
    return out


class TbStepProfiler:
    """One ``torch.profiler.step`` per training batch, writing TensorBoard traces.

    Construct after the ``SummaryWriter`` so traces land in the same ``tb/``
    directory the container's TensorBoard process already watches. ``close`` is
    safe to call more than once.
    """

    def __init__(
        self,
        cfg: DictConfig,
        log_dir: Path,
        device: torch.device,
        *,
        worker_name: str,
    ) -> None:
        self._prof = None
        opts = profiler_options(cfg)
        active = int(opts["active"])
        repeat = int(opts["repeat"])
        if not bool(opts["enabled"]) or active <= 0 or repeat <= 0:
            print("PyTorch profiler off (training.profiler.enabled=false or active/repeat is 0)")
            return
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = tensorboard_trace_handler(
            str(log_dir),
            worker_name=worker_name,
            use_gzip=True,
        )

        def _on_trace(prof: profile) -> None:
            handler(prof)
            print(f"PyTorch profiler trace written under {log_dir}")

        kwargs = {
            "activities": activities,
            "schedule": schedule(
                skip_first=max(0, int(opts["skip_first"])),
                wait=max(0, int(opts["wait"])),
                warmup=max(0, int(opts["warmup"])),
                active=active,
                repeat=repeat,
            ),
            "on_trace_ready": _on_trace,
            "record_shapes": bool(opts["record_shapes"]),
            "profile_memory": bool(opts["profile_memory"]),
            "with_stack": bool(opts["with_stack"]),
        }
        if "with_modules" in inspect.signature(profile).parameters:
            kwargs["with_modules"] = bool(opts["with_modules"])
        prof = profile(**kwargs)
        prof.__enter__()
        self._prof = prof
        recorded_after = (
            int(opts["skip_first"]) + int(opts["wait"]) + int(opts["warmup"]) + active
        )
        print(
            "PyTorch profiler on: "
            f"skip_first={int(opts['skip_first'])} wait={int(opts['wait'])} "
            f"warmup={int(opts['warmup'])} active={active} repeat={repeat} "
            f"(trace after about {recorded_after} batches) -> {log_dir}"
        )

    def step(self) -> None:
        """Advance the schedule. No-op when profiling is disabled or already closed."""
        if self._prof is not None:
            self._prof.step()

    def close(self) -> None:
        """Stop the profiler and flush a partial window if one is still recording."""
        prof = self._prof
        self._prof = None
        if prof is not None:
            prof.__exit__(None, None, None)
