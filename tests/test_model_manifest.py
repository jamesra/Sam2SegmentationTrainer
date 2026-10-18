from __future__ import annotations

import contextlib
import copy
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from annotation_crops import prompt_encoding as enc
from sam2_segmentation_trainer.model_manifest import (
    build_manifest,
    main,
    manifest_path,
    model_id_for_run,
    model_version_name,
    stem_mismatch,
    stem_model_version,
    validate_manifest,
    write_manifest,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "model_manifest_examples"


def _manifest() -> dict:
    return build_manifest(
        model_id="v4",
        model_version="v4.3",
        encoding=enc.encoding_metadata(0.95),
        training_data={"schema_version": 1, "set": "v3", "fingerprint": "abc"},
        run="sam2_em_v4",
        epoch=3,
    )


class ModelManifestTests(unittest.TestCase):
    def test_a_built_manifest_is_valid_and_carries_both_versions(self) -> None:
        manifest = _manifest()
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(manifest["model"]["model_version"], "v4.3")
        self.assertEqual(manifest["model"]["description"], "")  # the operator fills this in
        self.assertEqual(manifest["training_data"]["set"], "v3")  # the data version, apart from the model's
        self.assertEqual(manifest["server_contract"]["encoding"], enc.PROMPT_ENCODING_VERSION)
        self.assertEqual(manifest["client_contract"]["circle_diameter_scale"], 0.95)
        self.assertEqual(manifest["capabilities"], {"submission_mode": "tile_grid", "resolution_mode": "single"})

    def test_the_path_sits_next_to_the_weights_with_the_same_stem(self) -> None:
        self.assertEqual(manifest_path(Path("/m/sam2_em_v4_epoch3.pt")), Path("/m/sam2_em_v4_epoch3.model.json"))

    def test_a_misspelled_field_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["client_contract"]["requires_modelid"] = True
        manifest["capabilites"] = manifest.pop("capabilities")
        problems = validate_manifest(manifest)
        self.assertTrue(any("requires_modelid" in problem for problem in problems))
        self.assertTrue(any("capabilites" in problem for problem in problems))
        self.assertTrue(any("missing 'capabilities'" in problem for problem in problems))

    def test_bad_enum_values_are_rejected(self) -> None:
        manifest = _manifest()
        manifest["capabilities"]["submission_mode"] = "tiled"
        manifest["client_contract"]["mask_roles"] = ["seed", "negative"]
        manifest["server_contract"]["encoding"] = 3
        problems = validate_manifest(copy.deepcopy(manifest))
        self.assertEqual(len(problems), 3)

    def test_model_names_must_follow_the_convention(self) -> None:
        for model_id in ("sam2_em_v4", "4", "v4.3", "V4"):
            manifest = _manifest()
            manifest["model"]["model_id"] = model_id
            self.assertTrue(any("v<VERSION>" in problem for problem in validate_manifest(manifest)), model_id)
        for version in ("sam2_em_v4/epoch3", "v4", "v4.", "v4.x", "4.3"):
            manifest = _manifest()
            manifest["model"]["model_version"] = version
            self.assertTrue(any("v<VERSION>.<EPOCH>" in problem for problem in validate_manifest(manifest)), version)

    def test_the_description_is_text(self) -> None:
        manifest = _manifest()
        manifest["model"]["description"] = 3
        self.assertEqual(validate_manifest(manifest), ["model.description must be a string"])
        manifest["model"]["description"] = "Trained on the repopulated set."
        self.assertEqual(validate_manifest(manifest), [])

    def test_the_naming_helpers_follow_the_run_and_the_epoch(self) -> None:
        self.assertEqual(model_id_for_run("sam2_em_v12"), "v12")
        self.assertIsNone(model_id_for_run("scratch"))
        self.assertEqual(model_version_name("v4", 0), "v4.0")  # epochs count from 0, as the trainer logs them

    def test_a_non_object_is_rejected(self) -> None:
        self.assertEqual(validate_manifest([1, 2]), ["the manifest must be a JSON object"])

    def test_the_hand_written_v2_and_v3_examples_validate(self) -> None:
        names = sorted(path.name for path in EXAMPLES.glob("*.model.json"))
        self.assertEqual(names, ["sam2_em_v2.model.json", "sam2_em_v3.model.json"])
        for path in EXAMPLES.glob("*.model.json"):
            self.assertEqual(validate_manifest(json.loads(path.read_text(encoding="utf-8"))), [], path.name)
        v2 = json.loads((EXAMPLES / "sam2_em_v2.model.json").read_text(encoding="utf-8"))
        v3 = json.loads((EXAMPLES / "sam2_em_v3.model.json").read_text(encoding="utf-8"))
        self.assertEqual(v2["capabilities"]["submission_mode"], "full_viewport")
        self.assertEqual(v3["capabilities"]["submission_mode"], "tile_grid")
        self.assertEqual(v3["server_contract"]["encoding"], "legacy-flat-v1")
        self.assertEqual((v3["model"]["model_id"], v3["model"]["model_version"]), ("v3", "v3.9"))
        self.assertEqual((v2["model"]["model_id"], v2["model"]["model_version"]), ("v2", "v2.8"))
        for example in (v2, v3):
            self.assertIn("description", example["model"])

    def test_a_file_stem_may_lead_with_labels_joined_by_a_dash(self) -> None:
        for stem, version in (
            ("v1.5", "v1.5"), ("sem-v1.5", "v1.5"), ("tem-gpu_a-v4.3", "v4.3"), ("SEM-v12.10", "v12.10"),
        ):
            self.assertEqual(stem_model_version(stem), version, stem)
        # Not model names: left alone, because existing names stay as they are.
        for stem in ("serve_epoch5", "sam2_em_v3_epoch_9", "sem-v1", "sem_v1.5", "-v1.5", "sem--v1.5x", "v1.5-sem", ""):
            self.assertIsNone(stem_model_version(stem), stem)
            self.assertIsNone(stem_mismatch(stem, "v9.9"), stem)
        self.assertIsNone(stem_mismatch("sem-v1.5", "v1.5"))
        self.assertIn("v1.4", stem_mismatch("sem-v1.4", "v1.5"))
        self.assertEqual(manifest_path(Path("/m/sem-v1.5.pt")), Path("/m/sem-v1.5.model.json"))

    def test_the_check_command_flags_a_file_name_for_another_version(self) -> None:
        manifest = _manifest()  # model_version v4.3
        with TemporaryDirectory() as tmp:
            labelled = write_manifest(Path(tmp) / "sem-v4.3.model.json", manifest)
            legacy = write_manifest(Path(tmp) / "serve_epoch3.model.json", manifest)
            wrong = write_manifest(Path(tmp) / "sem-v4.2.model.json", manifest)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["check", str(labelled), str(legacy)]), 0)
                self.assertEqual(main(["check", str(wrong)]), 1)
        self.assertIn("says v4.2 but the model version is v4.3", out.getvalue())

    def test_the_check_command_reports_each_file(self) -> None:
        with TemporaryDirectory() as tmp:
            good = write_manifest(Path(tmp) / "good.model.json", _manifest())
            bad = Path(tmp) / "bad.model.json"
            bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["check", str(good)]), 0)
                self.assertEqual(main(["check", str(good), str(bad)]), 1)
        self.assertIn("INVALID", out.getvalue())
        self.assertIn("missing 'model'", out.getvalue())


if __name__ == "__main__":
    unittest.main()
