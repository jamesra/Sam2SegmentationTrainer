"""The shared prompt_encoders package: version lookup, and the old annotation_crops names are the same modules."""

import unittest

import prompt_encoders
from prompt_encoders import crush_display, v6


class PromptEncodersTests(unittest.TestCase):
    def test_v6_is_found_by_its_version(self):
        self.assertIs(prompt_encoders.for_version("edge-confidence-v6"), v6)
        self.assertEqual(prompt_encoders.versions(), ("edge-confidence-v6",))

    def test_an_unknown_version_names_the_known_ones(self):
        with self.assertRaisesRegex(KeyError, "edge-confidence-v6"):
            prompt_encoders.for_version("edge-confidence-v99")

    def test_the_annotation_crops_names_are_the_same_module_objects(self):
        from annotation_crops import crush_display as old_crush
        from annotation_crops import prompt_encoding as old_encoding

        self.assertIs(old_encoding, v6)
        self.assertIs(old_crush, crush_display)

    def test_from_imports_through_the_old_names_still_work(self):
        from annotation_crops.crush_display import display_lut
        from annotation_crops.prompt_encoding import PROMPT_ENCODING_VERSION, encode_mask_input

        self.assertIs(display_lut, crush_display.display_lut)
        self.assertIs(encode_mask_input, v6.encode_mask_input)
        self.assertEqual(PROMPT_ENCODING_VERSION, "edge-confidence-v6")


if __name__ == "__main__":
    unittest.main()
