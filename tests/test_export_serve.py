from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch import nn

from annotation_crops import prompt_encoding as enc
from sam2_segmentation_trainer.export_serve import (
    export_serve_checkpoint,
    finetuned_state_dict,
    launch_epoch_export,
    main,
    serve_metadata,
)


class ExportServeTests(unittest.TestCase):
    def test_raw_state_dict(self) -> None:
        raw = {"image_encoder.trunk.weight": 1, "sam_mask_decoder.bias": 2}
        self.assertIs(finetuned_state_dict(raw), raw)

    def test_meta_wrapper(self) -> None:
        inner = {"image_encoder.trunk.weight": 1}
        self.assertEqual(finetuned_state_dict({"model": inner}), inner)

    def test_rejects_non_dict(self) -> None:
        with self.assertRaises(TypeError):
            finetuned_state_dict([1, 2, 3])

    def test_rejects_unrecognized_dict(self) -> None:
        with self.assertRaises(ValueError):
            finetuned_state_dict({"optimizer": {}})

    def test_metadata_holds_every_encoding_constant(self) -> None:
        meta = serve_metadata(0.9)
        self.assertEqual(
            meta,
            {
                "prompt_encoding": enc.PROMPT_ENCODING_VERSION,
                "EDGE_TAU_PX": enc.EDGE_TAU_PX,
                "EDGE_TAU_DEPTH_FRAC": enc.EDGE_TAU_DEPTH_FRAC,
                "SEED_LOGIT": enc.SEED_LOGIT,
                "CORRECTION_LOGIT": enc.CORRECTION_LOGIT,
                "LOGIT_CLIP": enc.LOGIT_CLIP,
                "LOW_RES": enc.LOW_RES,
                "circle_diameter_scale": 0.9,
                "circle_annotation_trust": {"full_frac": 0.5, "zero_frac": 0.95, "ramp": "cosine"},
            },
        )
        self.assertIsInstance(meta["prompt_encoding"], str)
        self.assertEqual(serve_metadata()["circle_diameter_scale"], 0.95)

    def test_metadata_comes_from_the_runs_saved_config(self) -> None:
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "config.yaml").write_text(
                "prompt:\n  edge_tau_px: 6.0\n  seed_logit: 8.0\n  circle_diameter_scale: 0.9\n", encoding="utf-8"
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                meta = serve_metadata(run_dir=run)
        self.assertEqual(meta["EDGE_TAU_PX"], 6.0)
        self.assertEqual(meta["SEED_LOGIT"], 8.0)
        self.assertEqual(meta["circle_diameter_scale"], 0.9)
        self.assertEqual(meta["CORRECTION_LOGIT"], enc.CORRECTION_LOGIT)  # not in the config: the default
        self.assertEqual(meta["EDGE_TAU_DEPTH_FRAC"], enc.EDGE_TAU_DEPTH_FRAC)
        self.assertIn("WARNING: prompt.edge_tau_px is 6.0", out.getvalue())
        self.assertIn("WARNING: prompt.seed_logit is 8.0", out.getvalue())

    def test_a_run_with_default_values_exports_without_a_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "config.yaml").write_text("prompt:\n  edge_tau_px: 8.0\n  circle_diameter_scale: 0.95\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                serve_metadata(run_dir=run)
        self.assertEqual(out.getvalue(), "")

    def test_without_a_saved_config_the_defaults_are_used_with_a_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                meta = serve_metadata(run_dir=Path(tmp))
                missing = serve_metadata()
        self.assertEqual(meta["EDGE_TAU_PX"], enc.EDGE_TAU_PX)
        self.assertEqual(missing["SEED_LOGIT"], enc.SEED_LOGIT)
        self.assertEqual(out.getvalue().count("no saved config.yaml"), 2)

    def test_the_export_reads_the_config_next_to_the_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ), contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "config.yaml").write_text("prompt:\n  correction_logit: 4.0\n", encoding="utf-8")
            out = export_serve_checkpoint(
                finetuned=run / "best_model.pt", base_ckpt=Path("b.pt"), out=Path(tmp) / "serve.pt",
                model_id="v1", epoch=0,
            )
            payload = torch.load(out, weights_only=True)
        self.assertEqual(payload["prompt_encoding"]["CORRECTION_LOGIT"], 4.0)

    def test_the_export_writes_a_model_manifest_with_the_runs_data_record(self) -> None:
        from sam2_segmentation_trainer.model_manifest import validate_manifest

        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ), contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp) / "run_dir"
            run.mkdir()
            (run / "config.yaml").write_text("training:\n  run_name: sam2_em_v9\n", encoding="utf-8")
            (run / "training_data.json").write_text(json.dumps({"set": "v3", "fingerprint": "f00"}), encoding="utf-8")
            out = export_serve_checkpoint(
                finetuned=run / "best_model.pt", base_ckpt=Path("sam2.1.pt"), out=Path(tmp) / "x" / "serve_epoch2.pt",
                epoch=2,
            )
            manifest = json.loads((out.parent / "serve_epoch2.model.json").read_text(encoding="utf-8"))
            self.assertTrue((out.parent / "serve_epoch2.prompt_encoding.json").is_file())  # the older record stays
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(manifest["model"]["model_id"], "v9")
        self.assertEqual(manifest["model"]["model_version"], "v9.2")
        self.assertEqual(manifest["model"]["description"], "")
        self.assertEqual(manifest["training_data"], {"set": "v3", "fingerprint": "f00"})
        self.assertEqual(manifest["model"]["base_checkpoint"], "sam2.1.pt")

    def _export(self, tmp: str, **kwargs):
        with mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ), contextlib.redirect_stdout(io.StringIO()):
            return export_serve_checkpoint(
                finetuned=Path(tmp) / "run_v3" / "best_model.pt", base_ckpt=Path("b.pt"), out=Path(tmp) / "m.pt",
                model_id="v3", epoch=1, **kwargs,
            )

    def test_the_operators_description_survives_a_re_export(self) -> None:
        with TemporaryDirectory() as tmp:
            self._export(tmp)
            path = Path(tmp) / "m.model.json"
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["model"]["description"], "")
            written["model"]["description"] = "Trained on the repopulated set; use for RPC1."
            path.write_text(json.dumps(written), encoding="utf-8")

            self._export(tmp)  # the same export again
            kept = json.loads(path.read_text(encoding="utf-8"))["model"]["description"]
            self._export(tmp, description="A new description")  # an explicit one wins
            replaced = json.loads(path.read_text(encoding="utf-8"))["model"]["description"]
        self.assertEqual(kept, "Trained on the repopulated set; use for RPC1.")
        self.assertEqual(replaced, "A new description")

    def test_a_run_name_without_a_version_needs_an_explicit_model_id(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ) as load:
            with self.assertRaisesRegex(ValueError, "--model-id"):
                export_serve_checkpoint(
                    finetuned=Path(tmp) / "scratch" / "best_model.pt", base_ckpt=Path("b.pt"),
                    out=Path(tmp) / "m.pt", epoch=1,
                )
            with self.assertRaisesRegex(ValueError, "--model-id"):
                export_serve_checkpoint(
                    finetuned=Path(tmp) / "scratch" / "best_model.pt", base_ckpt=Path("b.pt"),
                    out=Path(tmp) / "m.pt", epoch=1, model_id="sam2_em_v4",
                )
            load.assert_not_called()  # refused before any weights are loaded

    def test_the_epoch_of_a_best_model_comes_from_the_file_the_trainer_wrote(self) -> None:
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "sam2_em_v4"
            run.mkdir()
            (run / "best_model.json").write_text(json.dumps({"epoch": 6, "global_step": 100}), encoding="utf-8")
            with mock.patch(
                "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
            ), contextlib.redirect_stdout(io.StringIO()):
                export_serve_checkpoint(finetuned=run / "best_model.pt", base_ckpt=Path("b.pt"), out=Path(tmp) / "m.pt")
            manifest = json.loads((Path(tmp) / "m.model.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model"]["model_version"], "v4.6")
        self.assertEqual(manifest["model"]["epoch"], 6)

    def test_an_unknown_epoch_is_refused_not_guessed(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ) as load:
            with self.assertRaisesRegex(ValueError, "--epoch"):
                export_serve_checkpoint(
                    finetuned=Path(tmp) / "sam2_em_v4" / "best_model.pt", base_ckpt=Path("b.pt"),
                    out=Path(tmp) / "m.pt",
                )
            load.assert_not_called()

    def test_the_command_line_reports_a_naming_problem_as_a_usage_error(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.paths.pretrained_checkpoint", return_value=Path("b.pt")
        ), contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as stop:
                main(["--checkpoint", str(Path(tmp) / "scratch" / "best_model.pt"), "--out", str(Path(tmp) / "m.pt")])
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("--model-id", err.getvalue())

    def test_a_run_without_a_data_record_exports_unknown_data(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ), contextlib.redirect_stdout(io.StringIO()):
            out = export_serve_checkpoint(
                finetuned=Path(tmp) / "old_run" / "best_model.pt", base_ckpt=Path("b.pt"), out=Path(tmp) / "s.pt",
                model_id="v7", epoch=5, display_name="Named",
            )
            manifest = json.loads((Path(tmp) / "s.model.json").read_text(encoding="utf-8"))
            self.assertTrue(out.is_file())
        self.assertEqual(manifest["training_data"]["set"], "unknown")
        self.assertEqual(manifest["model"]["model_id"], "v7")
        self.assertEqual(manifest["model"]["model_version"], "v7.5")
        self.assertEqual(manifest["model"]["display_name"], "Named")

    def test_the_exported_file_keeps_the_meta_layout_and_carries_the_metadata(self) -> None:
        model = nn.Linear(2, 2)
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=model
        ):
            out = export_serve_checkpoint(
                finetuned=Path("f.pt"), base_ckpt=Path("b.pt"), out=Path(tmp) / "nested" / "serve.pt",
                circle_diameter_scale=0.8, model_id="v1", epoch=0,
            )
            payload = torch.load(out, weights_only=True)
        self.assertEqual(set(payload), {"model", "prompt_encoding"})
        self.assertEqual(set(payload["model"]), set(model.state_dict()))
        self.assertEqual(payload["prompt_encoding"]["circle_diameter_scale"], 0.8)
        self.assertEqual(payload["prompt_encoding"]["SEED_LOGIT"], enc.SEED_LOGIT)
        # The loader still reads this file.
        self.assertEqual(finetuned_state_dict({"model": {"image_encoder.w": 1}, "prompt_encoding": {}}), {"image_encoder.w": 1})

    def test_the_epoch_export_runs_in_a_separate_detached_process(self) -> None:
        model = nn.Linear(2, 2)
        with TemporaryDirectory() as tmp, mock.patch("sam2_segmentation_trainer.export_serve.subprocess.Popen") as popen:
            run = Path(tmp)
            out = launch_epoch_export(model, run, 4)
            weights = run / "serve" / "serve_epoch4_weights.pt"
            saved = torch.load(weights, weights_only=True)
            (command,), kwargs = popen.call_args
            self.assertEqual(out, run / "serve" / "serve_epoch4.pt")
            self.assertEqual(set(saved), set(model.state_dict()))
        self.assertEqual(command[1:3], ["-m", "sam2_segmentation_trainer.export_serve"])
        self.assertEqual(command[command.index("--checkpoint") + 1], str(weights))
        self.assertEqual(command[command.index("--out") + 1], str(out))
        self.assertEqual(command[command.index("--run-dir") + 1], str(run))
        self.assertIn("--delete-checkpoint", command)
        self.assertEqual(command[command.index("--epoch") + 1], "4")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue("start_new_session" in kwargs or "creationflags" in kwargs)

    def test_delete_checkpoint_removes_the_weights_after_a_successful_export(self) -> None:
        with TemporaryDirectory() as tmp, mock.patch(
            "sam2_segmentation_trainer.export_serve.load_finetuned_model", return_value=nn.Linear(2, 2)
        ), mock.patch("sam2_segmentation_trainer.paths.pretrained_checkpoint", return_value=Path("b.pt")), (
            contextlib.redirect_stdout(io.StringIO())
        ):
            weights = Path(tmp) / "w.pt"
            weights.write_bytes(b"x")
            out = Path(tmp) / "serve.pt"
            main(["--checkpoint", str(weights), "--out", str(out), "--delete-checkpoint", "--model-id", "v1", "--epoch", "0"])
            self.assertTrue(out.is_file())
            self.assertFalse(weights.exists())
            kept = Path(tmp) / "kept.pt"
            kept.write_bytes(b"x")
            main(["--checkpoint", str(kept), "--out", str(Path(tmp) / "again.pt"), "--model-id", "v1", "--epoch", "0"])
            self.assertTrue(kept.exists())


if __name__ == "__main__":
    unittest.main()
