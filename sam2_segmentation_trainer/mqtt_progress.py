"""Publish SAM2 training progress onto the Nornir build dashboard.

Uses ``nornir_shared`` MQTT helpers when that package is installed. Every method
no-ops (console fallback for transcript lines) when the import or a publish fails,
so a missing broker never aborts training.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PIPELINE = "sam2-em-train"
_RUN_ID_ENV = "NORNIR_RUN_ID"
_LOAD_PUBLISHERS = object()


def format_batch_samples(
    meta: Sequence[Mapping[str, Any]],
    *,
    max_samples: int = 8,
) -> str:
    """Compact ``volume/image_key#location_id`` list for warning transcripts."""
    parts: list[str] = []
    for item in meta[:max_samples]:
        volume = item.get("volume", "?")
        image_key = item.get("image_key", "?")
        location_id = item.get("location_id", "?")
        parts.append(f"{volume}/{image_key}#{location_id}")
    text = ", ".join(parts) if parts else "(no meta)"
    extra = len(meta) - max_samples
    if extra > 0:
        text = f"{text} (+{extra} more)"
    return text


@dataclass
class StepAnomalyMonitor:
    """Warn on training steps that are much slower or higher-loss than recent medians.

    History is updated after the check so the current outlier does not inflate the
    median used for that step. Warmup steps only fill history.
    """

    warmup: int = 30
    window: int = 100
    slow_factor: float = 2.5
    slow_seconds: float = 3.0
    high_loss_factor: float = 5.0
    high_loss_absolute: float = 3.0
    _times: deque[float] = field(default_factory=deque, init=False, repr=False)
    _losses: deque[float] = field(default_factory=deque, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._times = deque(maxlen=max(1, int(self.window)))
        self._losses = deque(maxlen=max(1, int(self.window)))

    def observe(
        self,
        *,
        progress: TrainingProgress,
        epoch: int,
        step: int,
        duration_s: float,
        loss: float,
        meta: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        """Compare against history, emit MQTT warnings, then record this step.

        Returns the warning messages that were published (empty when none).
        """
        self._seen += 1
        messages: list[str] = []
        samples = format_batch_samples(meta)
        ready = self._seen > int(self.warmup) and len(self._times) >= 10

        if ready:
            med_t = float(statistics.median(self._times))
            threshold_t = max(float(self.slow_seconds), med_t * float(self.slow_factor))
            if float(duration_s) >= threshold_t:
                msg = (
                    f"slow batch epoch={epoch} step={step} "
                    f"took={float(duration_s):.2f}s (median={med_t:.2f}s) "
                    f"loss={float(loss):.4f} samples={samples}"
                )
                progress.warn(msg)
                messages.append(msg)

            med_l = float(statistics.median(self._losses))
            threshold_l = max(
                float(self.high_loss_absolute),
                med_l * float(self.high_loss_factor),
            )
            if float(loss) >= threshold_l:
                msg = (
                    f"high loss epoch={epoch} step={step} "
                    f"loss={float(loss):.4f} (median={med_l:.4f}) "
                    f"took={float(duration_s):.2f}s samples={samples}"
                )
                progress.warn(msg)
                messages.append(msg)

        self._times.append(float(duration_s))
        self._losses.append(float(loss))
        return messages


@dataclass
class _Publishers:
    early: Callable[..., None]
    meta: Callable[..., None]
    event: Callable[..., None]
    log: Callable[..., None]
    log_err: Callable[..., None]


def _load_publishers() -> _Publishers | None:
    """Import Nornir MQTT helpers, or return None when the package is absent."""
    try:
        from nornir_shared.mqtt_telemetry import (
            publish_early_run_meta,
            publish_run_event,
            publish_run_meta,
        )
        from nornir_shared.prettyoutput import Log, LogErr
    except ImportError:
        return None
    return _Publishers(
        early=publish_early_run_meta,
        meta=publish_run_meta,
        event=publish_run_event,
        log=Log,
        log_err=LogErr,
    )


class TrainingProgress:
    """One training run's dashboard publications.

    ``status`` stays ``completed`` until the training loop marks a stop or an
    exception is recorded. ``close`` publishes that terminal status once.
    """

    status: str
    error: str | None
    started: bool
    _publishers: _Publishers | None
    _closed: bool

    def __init__(self, publishers: _Publishers | None | object = _LOAD_PUBLISHERS) -> None:
        """Bind publishers. The default loads ``nornir_shared``; pass None to disable.

        Tests pass an explicit ``_Publishers`` instance.
        """
        self.status = "completed"
        self.error = None
        self.started = False
        self._closed = False
        if publishers is _LOAD_PUBLISHERS:
            self._publishers = _load_publishers()
        elif isinstance(publishers, _Publishers):
            self._publishers = publishers
        else:
            self._publishers = None

    def start(self, *, run_name: str, volumepath: str, compute: str = "cuda") -> None:
        """Publish retained run meta and open the train stage.

        Sets ``NORNIR_RUN_ID`` to ``run_name`` so a resume updates the same
        dashboard row.
        """
        os.environ[_RUN_ID_ENV] = run_name
        self.started = True
        self._call(
            None if self._publishers is None else self._publishers.early,
            pipeline=PIPELINE,
            volumepath=volumepath,
            status="running",
            compute=compute,
        )
        self.stage("train", "start")

    def stage(self, function: str, phase: str) -> None:
        """Publish ``stage_start`` or ``stage_end`` so the dashboard current stage updates."""
        event = "stage_start" if phase == "start" else "stage_end"
        self._event(
            event,
            module="sam2_segmentation_trainer",
            function=function,
        )

    def report_epoch(self, epoch: int, num_epochs: int) -> None:
        """Publish the epochs bar. ``epoch`` is 0-based; the bar shows ``epoch + 1``."""
        self._event(
            "iterate_progress",
            track_id="epochs",
            label="epochs",
            current=int(epoch) + 1,
            total=int(num_epochs),
            depth=0,
        )

    def report_step(
        self,
        *,
        step: int,
        batches_per_epoch: int,
        global_step: int,
        log_every: int,
        loss: float,
        lr: float,
    ) -> None:
        """Publish the steps bar and an info line on the TensorBoard log cadence.

        ``step`` is the 0-based index within the epoch. No publish when
        ``log_every`` is less than 1 or ``global_step`` is off the interval.
        """
        every = int(log_every)
        if every <= 0 or int(global_step) % every != 0:
            return
        label = f"steps  loss {loss:.4f}  lr {lr:.1e}"
        self._event(
            "iterate_progress",
            track_id="steps",
            label=label,
            current=int(step) + 1,
            total=int(batches_per_epoch),
            depth=1,
        )
        self.transcript(
            f"step {int(step) + 1}/{int(batches_per_epoch)} loss={loss:.4f} lr={lr:.6g}"
        )

    def transcript(self, message: str) -> None:
        """Info log on the dashboard transcript. Prints when MQTT logging is unavailable."""
        self._info(message)

    def warn(self, message: str) -> None:
        """Warning on the dashboard transcript (LogErr when MQTT is available)."""
        text = message if message.startswith("WARN") else f"WARN {message}"
        self._error(text)

    def log_error(self, message: str) -> None:
        """Error on the dashboard transcript. Does not fail the run."""
        self._error(message)

    def mark_stopped(self) -> None:
        """Record a user interrupt. Does not override a failure already recorded."""
        if self.status == "completed":
            self.status = "stopped"

    def mark_failed(self, error: str) -> None:
        """Record a failure. Does not override an interrupt already recorded."""
        if self.status != "completed":
            return
        self.status = "failed"
        self.error = error

    def close(self) -> None:
        """Publish the terminal status once, after the training loop returns or raises."""
        if self._closed:
            return
        self._closed = True
        if not self.started:
            return
        self.finish(self.status, error=self.error)

    def finish(self, status: str, *, error: str | None = None) -> None:
        """Publish terminal run meta. ``completed`` also closes the train stage."""
        if status == "completed":
            self.stage("train", "end")
        if error:
            self._error(error)
        self._call(
            None if self._publishers is None else self._publishers.meta,
            status=status,
            end_ts=time.time(),
        )

    def _event(self, event: str, **fields: Any) -> None:
        publisher = None if self._publishers is None else self._publishers.event
        self._call(publisher, event, **fields)

    def _info(self, message: str) -> None:
        if self._publishers is None:
            print(message)
            return
        try:
            self._publishers.log(message)
        except Exception:
            print(message)

    def _error(self, message: str) -> None:
        if self._publishers is None:
            print(message)
            return
        try:
            self._publishers.log_err(message)
        except Exception:
            print(message)

    @staticmethod
    def _call(fn: Callable[..., None] | None, *args: Any, **kwargs: Any) -> None:
        if fn is None:
            return
        try:
            fn(*args, **kwargs)
        except Exception:
            return
