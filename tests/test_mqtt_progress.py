"""Training progress publications, without a broker."""

from __future__ import annotations

import builtins
import os
import unittest
from io import StringIO
from unittest.mock import Mock, patch

from sam2_segmentation_trainer.mqtt_progress import (
    PIPELINE,
    TrainingProgress,
    _Publishers,
    _load_publishers,
)


class MqttProgressTests(unittest.TestCase):
    def test_epoch_and_step_payloads_follow_log_cadence(self) -> None:
        event = Mock()
        log = Mock()
        meta = Mock()
        early = Mock()
        publishers = _Publishers(
            early=early,
            meta=meta,
            event=event,
            log=log,
            log_err=Mock(),
        )
        progress = TrainingProgress(publishers=publishers)
        progress.start(run_name="sam2_em_v3", volumepath="/data-local")
        self.assertEqual(os.environ["NORNIR_RUN_ID"], "sam2_em_v3")
        early.assert_called_once_with(
            pipeline=PIPELINE,
            volumepath="/data-local",
            status="running",
            compute="cuda",
        )
        event.assert_any_call(
            "stage_start",
            module="sam2_segmentation_trainer",
            function="train",
        )

        progress.report_epoch(2, 15)
        event.assert_any_call(
            "iterate_progress",
            track_id="epochs",
            label="epochs",
            current=3,
            total=15,
            depth=0,
        )

        event.reset_mock()
        log.reset_mock()
        progress.report_step(
            step=4,
            batches_per_epoch=100,
            global_step=1,
            log_every=50,
            loss=0.5,
            lr=1e-4,
        )
        event.assert_not_called()
        log.assert_not_called()

        progress.report_step(
            step=0,
            batches_per_epoch=100,
            global_step=0,
            log_every=50,
            loss=0.421,
            lr=1e-5,
        )
        progress.report_step(
            step=49,
            batches_per_epoch=100,
            global_step=50,
            log_every=50,
            loss=0.2,
            lr=1e-5,
        )
        self.assertEqual(event.call_count, 2)
        event.assert_any_call(
            "iterate_progress",
            track_id="steps",
            label="steps  loss 0.4210  lr 1.0e-05",
            current=1,
            total=100,
            depth=1,
        )
        self.assertEqual(log.call_count, 2)
        self.assertIn("loss=0.4210", log.call_args_list[0].args[0])

        progress.mark_stopped()
        progress.close()
        meta.assert_called_once()
        self.assertEqual(meta.call_args.kwargs["status"], "stopped")
        self.assertIn("end_ts", meta.call_args.kwargs)
        progress.close()
        meta.assert_called_once()

    def test_missing_nornir_does_not_raise(self) -> None:
        with patch(
            "sam2_segmentation_trainer.mqtt_progress._load_publishers",
            return_value=None,
        ):
            progress = TrainingProgress()
        stdout = StringIO()
        with patch("sys.stdout", stdout):
            progress.start(run_name="sam2_em_v1", volumepath="/data")
            progress.report_epoch(0, 2)
            progress.report_step(
                step=0,
                batches_per_epoch=10,
                global_step=0,
                log_every=50,
                loss=1.25,
                lr=1e-4,
            )
            progress.report_step(
                step=1,
                batches_per_epoch=10,
                global_step=1,
                log_every=50,
                loss=1.0,
                lr=1e-4,
            )
            progress.transcript("epoch 0 val IoU 0.5000")
            progress.close()
        text = stdout.getvalue()
        self.assertIn("loss=1.2500", text)
        self.assertIn("epoch 0 val IoU 0.5000", text)
        self.assertEqual(text.count("loss="), 1)
        self.assertEqual(os.environ["NORNIR_RUN_ID"], "sam2_em_v1")

    def test_import_error_disables_publishers(self) -> None:
        real_import = builtins.__import__

        def _import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("nornir_shared"):
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_import):
            self.assertIsNone(_load_publishers())

    def test_publisher_exception_is_swallowed(self) -> None:
        publishers = _Publishers(
            early=Mock(side_effect=RuntimeError("broker down")),
            meta=Mock(side_effect=RuntimeError("broker down")),
            event=Mock(side_effect=RuntimeError("broker down")),
            log=Mock(side_effect=RuntimeError("broker down")),
            log_err=Mock(),
        )
        progress = TrainingProgress(publishers=publishers)
        stdout = StringIO()
        with patch("sys.stdout", stdout):
            progress.start(run_name="sam2_em_v2", volumepath="/data")
            progress.report_epoch(0, 1)
            progress.report_step(
                step=0,
                batches_per_epoch=4,
                global_step=0,
                log_every=1,
                loss=0.1,
                lr=1e-3,
            )
            progress.mark_failed("RuntimeError: boom")
            progress.close()
        self.assertEqual(progress.status, "failed")
        self.assertIn("loss=0.1000", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
