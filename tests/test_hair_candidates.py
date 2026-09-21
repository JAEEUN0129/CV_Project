"""Regression checks for reference conditioning and recolouring boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from jaeeun.vace.color_candidates import (
    build_color_candidates, hair_mask_from_scores, recolor_requested,
)
from jaeeun.vace.style_options import hairstyle_prompt


class HairCandidatesTest(unittest.TestCase):
    def test_connected_tips_without_skin_or_detached_noise(self):
        classes = np.zeros((100, 100), dtype=np.uint8)
        classes[20:60, 20:60] = 1
        probabilities = np.zeros((3, 100, 100), dtype=np.float32)
        probabilities[0] = 0.9
        probabilities[1, 20:60, 20:60] = 0.95
        probabilities[1, 60:62, 30:40] = 0.3  # uncertain tips
        probabilities[1, 61:63, 61:63] = 0.3  # detached nearby noise
        probabilities[1, 30:40, 60:62] = 0.3
        probabilities[2, 30:40, 60:62] = 0.7  # protected skin
        mask = hair_mask_from_scores(classes, {0: "background", 1: "hair", 2: "skin"}, probabilities)
        self.assertTrue(mask[60:62, 30:40].all())
        self.assertFalse(mask[61:63, 61:63].any())
        self.assertFalse(mask[30:40, 60:62].any())

    def test_narrow_strand_is_coloured_without_changing_background(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            output = Path(directory) / "output.png"
            original = np.full((30, 30, 3), [220, 180, 100], dtype=np.uint8)
            Image.fromarray(original).save(source)
            mask = np.zeros((30, 30), dtype=bool)
            mask[5:25, 15] = True
            recolor_requested(source, mask, "#202A3A", output)
            result = np.array(Image.open(output))
            np.testing.assert_array_equal(result[~mask], original[~mask])
            self.assertLess(float(result[mask].mean()), 100)

    def test_reference_bangs_survive_wave_refinement_contract(self):
        options = {"length": "장발", "wave": "C컬"}
        prompt = hairstyle_prompt(options, True)
        self.assertIn("Copy the bangs from the reference image", prompt)
        self.assertIn("Do not create S-waves", prompt)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, mask = (root / name for name in ("source.png", "ref.png", "mask.png"))
            Image.new("RGB", (40, 60), "gray").save(source)
            Image.new("RGB", (40, 60), "gray").save(reference)
            Image.new("L", (40, 60), 255).save(mask)

            def generate(src, ref, edit_mask, text, output):
                Image.new("RGB", (40, 60), "gray").save(output)

            with patch("jaeeun.vace.color_candidates.FluxAnchorEditor.create", side_effect=generate) as editor, patch(
                "jaeeun.vace.color_candidates.requested_hair_mask", return_value=np.ones((60, 40), dtype=bool)
            ):
                result = build_color_candidates(source, reference, mask, root, "summer_cool",
                                                Path("python"), Path("worker"), prompt,
                                                include_requested=True, requested_options=options)
            self.assertEqual(len(result), 4)
            self.assertEqual(editor.call_count, 2)  # recommendations do not regenerate hair
            correction = editor.call_args_list[1].args
            self.assertEqual(correction[1], reference)
            self.assertIn("Copy the bangs from the reference image", correction[3])
            explicit = hairstyle_prompt({**options, "bangs": "커튼뱅"}, True)
            self.assertNotIn("Copy the bangs from the reference image", explicit)


if __name__ == "__main__":
    unittest.main()
