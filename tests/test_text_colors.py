import unittest

import numpy as np
from PIL import Image, ImageDraw

import img2pptx


class TextColorAnalysisTests(unittest.TestCase):
    def make_line(self, text, secondary_count=0):
        """Render a deterministic, single OCR line with per-character boxes."""
        font = img2pptx.ref_font(100)
        image = Image.new("RGB", (1000, 180), "white")
        draw = ImageDraw.Draw(image)
        baseline = 135
        x = 40
        char_boxes = []
        glyph_boxes = []
        primary = (15, 42, 112)
        secondary = (245, 155, 72)

        for i, char in enumerate(text):
            advance = font.getlength(char)
            is_secondary = i >= len(text) - secondary_count
            color = secondary if is_secondary else primary
            box = font.getbbox(char, anchor="ls")
            draw.text((round(x), baseline), char, font=font, fill=color, anchor="ls")
            char_boxes.append((round(x), round(x + advance)))
            glyph_boxes.append((round(x + box[0]), baseline + box[1],
                                round(x + box[2]), baseline + box[3]))
            x += advance

        x0 = min(box[0] for box in glyph_boxes)
        y0 = min(box[1] for box in glyph_boxes)
        x1 = max(box[2] for box in glyph_boxes)
        y1 = max(box[3] for box in glyph_boxes)
        rgb = np.asarray(image)
        secondary_mask = np.all(rgb == secondary, axis=2)
        line = img2pptx.Line(text, x0, y0, x1 - x0, y1 - y0, chars=char_boxes)
        return rgb[:, :, ::-1].copy(), line, secondary_mask

    def analyze(self, text, secondary_count=0):
        image, line, secondary_mask = self.make_line(text, secondary_count)
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.assertTrue(img2pptx.analyze_line(image, line, full_mask))
        return line, full_mask, secondary_mask

    def test_minority_lighter_color_keeps_ink_and_splits_color_runs(self):
        line, full_mask, secondary_pixels = self.analyze("ABCDEFGHJK", secondary_count=2)

        # All solid pixels of the two light orange glyphs must survive color filtering.
        self.assertGreater(int(secondary_pixels.sum()), 100)
        self.assertTrue(np.all(full_mask[secondary_pixels] > 0))
        self.assertEqual([run[0] for run in line.runs], ["ABCDEFGH", "JK"])
        self.assertGreater(np.linalg.norm(np.array(line.runs[0][1]) - line.runs[1][1]), 45)

    def test_single_color_line_stays_one_run(self):
        line, _, _ = self.analyze("ABCDEFGHJK")

        self.assertEqual(len(line.runs), 1)
        self.assertEqual(line.runs[0][0], line.text)

    def test_latin_word_strokes_are_not_removed_as_decoration(self):
        image, line, _ = self.make_line("KPI")
        img2pptx.strip_side_bars(image, [line])
        self.assertEqual(line.text, "KPI")

    def test_single_letter_box_can_exclude_an_unrelated_chart(self):
        image = Image.new("RGB", (200, 200), (255, 90, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 25, 60, 60), fill="white")
        draw.rectangle((65, 10, 85, 60), fill="white")
        draw.text((40, 95), "A", font=img2pptx.ref_font(60, True), fill="white", anchor="lt")
        line = img2pptx.Line("A", 30, 5, 90, 160)
        self.assertTrue(img2pptx._refine_short_glyph(np.asarray(image)[:, :, ::-1].copy(), line))
        self.assertGreater(line.y, 80)
        self.assertLess(line.h, 60)


if __name__ == "__main__":
    unittest.main()
