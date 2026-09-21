"""Regression checks for reference conditioning and recolouring boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from jaeeun.vace.color_candidates import (
    build_color_candidates, hair_mask_from_scores, recolor_requested, complete_source_edit_mask,
)
from jaeeun.vace.style_options import hairstyle_prompt
from jaeeun.vace.regional_edit import composite_region, regional_masks
from jaeeun.vace.color_candidates import merge_detail_mask


class HairCandidatesTest(unittest.TestCase):
    def test_reference_only_passes_reference_to_every_preview_without_postprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("jaeeun.vace.color_candidates.FluxAnchorEditor.create") as editor, patch(
                "jaeeun.vace.color_candidates.complete_source_edit_mask"
            ) as complete, patch("jaeeun.vace.color_candidates.regional_masks") as regions, patch(
                "jaeeun.vace.color_candidates.requested_hair_mask"
            ) as segment, patch("jaeeun.vace.color_candidates.recolor_requested") as recolor:
                result = build_color_candidates(root / "source.png", root / "ref.png", root / "mask.png", root,
                    "summer_cool", Path("python"), Path("worker"), hairstyle_prompt({}, True),
                    include_requested=True, requested_options={"bangs": None, "wave": None})
            self.assertEqual(len(result), 4)
            self.assertEqual(editor.call_count, 4)
            for call in editor.call_args_list:
                self.assertEqual(call.args[:3], (root / "source.png", root / "ref.png", root / "mask.png"))
            self.assertIn("Match the hair color of the hairstyle reference image", editor.call_args_list[0].args[3])
            complete.assert_not_called()
            regions.assert_not_called()
            segment.assert_not_called()
            recolor.assert_not_called()

    def test_text_only_uses_original_single_pass_generation_for_each_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = {"bangs": "커튼뱅", "length": "장발", "wave": "C컬"}
            with patch("jaeeun.vace.color_candidates.FluxAnchorEditor.create") as editor, patch(
                "jaeeun.vace.color_candidates.complete_source_edit_mask"
            ) as complete, patch("jaeeun.vace.color_candidates.regional_masks") as regions, patch(
                "jaeeun.vace.color_candidates.requested_hair_mask"
            ) as segment, patch("jaeeun.vace.color_candidates.recolor_requested") as recolor:
                result = build_color_candidates(root / "source.png", None, root / "mask.png", root,
                    "summer_cool", Path("python"), Path("worker"), hairstyle_prompt(options, False),
                    include_requested=True, requested_color="브라운", requested_options=options)
            self.assertEqual(len(result), 4)
            self.assertEqual(editor.call_count, 4)
            for call in editor.call_args_list:
                self.assertEqual(call.args[:3], (root / "source.png", None, root / "mask.png"))
                self.assertIn("C-shaped", call.args[3])
            self.assertIn("natural brown", editor.call_args_list[0].args[3])
            complete.assert_not_called()
            regions.assert_not_called()
            segment.assert_not_called()
            recolor.assert_not_called()

    def test_recolor_preserves_shadows_and_desaturates_reflections(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.full((80, 160, 3), 110, dtype=np.uint8)
            pixels[:, :40] = 4
            pixels[:, 120:] = 245
            source = root / "source.png"
            Image.fromarray(pixels).save(source)
            recolor_requested(source, np.ones((80, 160), dtype=bool), "#9A6F78", root / "color.png")
            result = np.array(Image.open(root / "color.png"))
            lab = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            self.assertLess(lab[40, 20, 0], 5)
            self.assertLess(lab[40, 80, 0], lab[40, 140, 0])
            self.assertLess(np.linalg.norm(lab[40, 140, 1:]), np.linalg.norm(lab[40, 80, 1:]) * .5)

    def test_original_hair_tips_are_editable_but_not_added_to_recolor_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_hair = np.zeros((60, 40), dtype=bool)
            old_hair[10:55, 10:15] = True
            initial = np.zeros_like(old_hair)
            initial[10:30, 10:25] = True
            Image.fromarray(initial.astype(np.uint8) * 255).save(root / "edit.png")
            with patch("jaeeun.vace.color_candidates.requested_hair_mask", return_value=old_hair):
                result = complete_source_edit_mask(root / "source.png", root / "edit.png", root / "complete.png")
            completed = np.array(Image.open(result)) > 0
            self.assertTrue(completed[30:55, 10:15].all())
            np.testing.assert_array_equal(completed, old_hair | initial)
            # Recolouring only uses final-image segmentation, not this edit mask.
            source = root / "requested.png"
            Image.new("RGB", (40, 60), "gray").save(source)
            recolor_requested(source, initial, "#202A3A", root / "color.png")
            self.assertTrue((np.array(Image.open(root / "color.png"))[40, 12] == [128, 128, 128]).all())

    def test_curtain_bangs_open_temples_without_editing_eyes(self):
        classes = np.ones((100, 100), dtype=np.uint8)
        classes[30:80, 30:70] = 2
        classes[45:48, 35:65] = 3
        classes[50:53, 35:65] = 4
        classes[45:50, 48:52] = 1  # old fringe below the highest brow
        labels = {0: "background", 1: "hair", 2: "skin", 3: "l_brow", 4: "l_eye"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (100, 100), "gray").save(source)
            with patch("jaeeun.vace.regional_edit.HumanParser") as parser:
                parser.return_value.predict.return_value = (classes, labels)
                curtain, body = regional_masks(source, "커튼뱅")
                choppy, _ = regional_masks(source, "처피뱅")
            self.assertTrue(curtain[55, 29])
            self.assertFalse(choppy[55, 29])
            self.assertTrue(curtain[45:50, 48:52].all())
            self.assertFalse(curtain[classes == 3].any())
            self.assertFalse(curtain[classes == 4].any())
            self.assertFalse((curtain & body).any())

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

            fringe = np.zeros((60, 40), dtype=bool)
            fringe[:20] = True
            with patch("jaeeun.vace.color_candidates.FluxAnchorEditor.create", side_effect=generate) as editor, patch(
                "jaeeun.vace.color_candidates.requested_hair_mask", return_value=np.ones((60, 40), dtype=bool)
            ), patch("jaeeun.vace.color_candidates.regional_masks", return_value=(fringe, ~fringe)):
                result = build_color_candidates(source, reference, mask, root, "summer_cool",
                                                Path("python"), Path("worker"), prompt,
                                                include_requested=True, requested_options=options)
            self.assertEqual(len(result), 4)
            self.assertEqual(editor.call_count, 2)  # recommendations do not regenerate hair
            correction = editor.call_args_list[1].args
            self.assertEqual(editor.call_args_list[0].args[1], reference)
            self.assertIsNone(correction[1])
            self.assertIn("locked", correction[3])
            self.assertFalse(np.array(Image.open(correction[2]))[:20].any())
            explicit = hairstyle_prompt({**options, "bangs": "커튼뱅"}, True)
            self.assertNotIn("Copy the bangs from the reference image", explicit)

    def test_region_composite_restores_protected_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (30, 40), "red").save(root / "source.png")
            Image.new("RGB", (30, 40), "blue").save(root / "edited.png")
            mask = np.zeros((40, 30), dtype=bool)
            mask[20:, :5] = True
            composite_region(root / "source.png", root / "edited.png", mask, root / "out.png")
            result = np.array(Image.open(root / "out.png"))
            self.assertTrue((result[~mask] == [255, 0, 0]).all())
            self.assertTrue((result[mask] == [0, 0, 255]).all())

    def test_explicit_bangs_are_corrected_after_body_without_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (40, 60), "gray").save(source)
            fringe = np.zeros((60, 40), dtype=bool)
            fringe[:20] = True
            def generate(src, ref, mask, prompt, output):
                Image.new("RGB", (40, 60), "gray").save(output)
            options = {"bangs": "커튼뱅", "length": "단발", "wave": "생머리"}
            with patch("jaeeun.vace.color_candidates.FluxAnchorEditor.create", side_effect=generate) as editor, patch(
                "jaeeun.vace.color_candidates.regional_masks", return_value=(fringe, ~fringe)
            ), patch("jaeeun.vace.color_candidates.requested_hair_mask", return_value=np.ones((60, 40), dtype=bool)):
                build_color_candidates(source, source, source, root, "summer_cool",
                                       Path("python"), Path("worker"), hairstyle_prompt(options, True),
                                       include_requested=True, requested_options=options)
            self.assertEqual(editor.call_count, 3)
            body, bangs = [call.args for call in editor.call_args_list[1:]]
            self.assertIsNone(body[1])
            self.assertIsNone(bangs[1])
            self.assertEqual(bangs[0].name, "anchor-body.png")
            self.assertIn("connect continuously", bangs[3])
            self.assertTrue(np.array(Image.open(bangs[2]))[:20].all())

    def test_detail_mask_only_adds_connected_hair_and_protects_skin(self):
        base = np.zeros((20, 20), dtype=bool)
        base[:8, :8] = True
        detail = np.zeros_like(base)
        detail[7:15, :8] = True
        detail[17:, 17:] = True
        classes = np.zeros_like(base, dtype=np.uint8)
        classes[13:15, :8] = 1
        result = merge_detail_mask(base, detail, classes, {0: "background", 1: "skin"})
        self.assertTrue(result[10, 2])
        self.assertFalse(result[14, 2])
        self.assertFalse(result[18, 18])


if __name__ == "__main__":
    unittest.main()
