import unittest
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np

from arrow_graphics import detect_graphic_arrows


def draw_arrow(image, box, direction, color):
    x, y, w, h = box
    polygons = {
        "up": [(x + .35 * w, y + h), (x + .35 * w, y + .38 * h),
               (x + .08 * w, y + .38 * h), (x + .5 * w, y),
               (x + .92 * w, y + .38 * h), (x + .65 * w, y + .38 * h),
               (x + .65 * w, y + h)],
        "right": [(x, y + .35 * h), (x + .62 * w, y + .35 * h),
                  (x + .62 * w, y + .08 * h), (x + w, y + .5 * h),
                  (x + .62 * w, y + .92 * h), (x + .62 * w, y + .65 * h),
                  (x, y + .65 * h)],
    }
    polygon = np.round(np.asarray(polygons[direction])).astype(np.int32)
    cv2.fillPoly(image, [polygon], color)


class ArrowGraphicTests(unittest.TestCase):
    def test_tall_up_arrow_misread_as_ge_keeps_label(self):
        image = np.full((300, 600, 3), 255, np.uint8)
        draw_arrow(image, (350, 130, 28, 48), 'up', (0, 120, 245))
        line = SimpleNamespace(text='总成本个', x=242, y=130, w=144, h=48, chars=None)
        lines, arrows = detect_graphic_arrows(image, [line])
        self.assertEqual([item.text for item in lines], ['总成本'])
        self.assertEqual(len(arrows), 1)


    def test_gray_flow_arrow_is_returned_as_a_local_mask(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        draw_arrow(image, (35, 42, 60, 30), "right", (220, 200, 180))

        lines, arrows = detect_graphic_arrows(image)

        self.assertEqual(lines, [])
        self.assertEqual(len(arrows), 1)
        arrow = arrows[0]
        self.assertEqual(arrow.direction, "right")
        self.assertEqual(arrow.color, "gray")
        self.assertEqual(arrow.box, (35, 44, 61, 27))
        self.assertEqual(arrow.mask.shape, (27, 61))
        self.assertGreater(arrow.score, .5)

    def test_edge_arrow_glyph_is_trimmed_without_dropping_chinese_label(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        draw_arrow(image, (75, 24, 26, 50), "up", (0, 112, 255))
        line = SimpleNamespace(text="库存↑", x=20, y=24, w=81, h=50,
                               chars=[(20, 45), (45, 70), (75, 101)])

        filtered, arrows = detect_graphic_arrows(image, [line])

        self.assertEqual(len(arrows), 1)
        self.assertEqual(filtered[0].text, "库存")
        self.assertEqual(filtered[0].x, 20)
        self.assertEqual(filtered[0].w, 55)
        self.assertEqual(line.text, "库存↑")  # Input OCR records are not mutated.
        self.assertEqual(arrows[0].text_changes[0].removed_text, "↑")

    def test_arrow_only_ocr_line_is_removed(self):
        image = np.full((120, 140, 3), 255, dtype=np.uint8)
        draw_arrow(image, (75, 24, 26, 50), "up", (0, 112, 255))
        line = SimpleNamespace(text="↑", x=75, y=24, w=26, h=50,
                               chars=[(75, 101)])

        filtered, arrows = detect_graphic_arrows(image, [line])

        self.assertEqual(len(arrows), 1)
        self.assertEqual(filtered, [])

    def test_misrecognized_dash_suffix_is_trimmed(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        draw_arrow(image, (75, 24, 26, 50), "up", (0, 112, 255))
        line = SimpleNamespace(text="库存一", x=20, y=24, w=81, h=50,
                               chars=[(20, 45), (45, 70), (75, 101)])

        filtered, arrows = detect_graphic_arrows(image, [line])

        self.assertEqual(len(arrows), 1)
        self.assertEqual(filtered[0].text, "库存")

    def test_inline_numeric_arrow_stays_editable_text(self):
        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        draw_arrow(image, (75, 43, 26, 30), "right", (0, 112, 255))
        line = SimpleNamespace(text="75%→80%", x=20, y=42, w=115, h=32,
                               chars=[(20, 32), (32, 44), (44, 56), (56, 82),
                                      (82, 94), (94, 106), (106, 118)])

        filtered, arrows = detect_graphic_arrows(image, [line])

        self.assertEqual(arrows, [])
        self.assertEqual(filtered[0].text, "75%→80%")

    def test_inline_arrow_at_another_vertical_position_does_not_block_icon(self):
        image = np.full((140, 180, 3), 255, dtype=np.uint8)
        draw_arrow(image, (75, 82, 60, 30), "right", (0, 112, 255))
        line = SimpleNamespace(text="75%→80%", x=20, y=20, w=115, h=28,
                               chars=[(20, 32), (32, 44), (44, 56), (56, 82),
                                      (82, 94), (94, 106), (106, 118)])

        filtered, arrows = detect_graphic_arrows(image, [line])

        self.assertEqual(len(arrows), 1)
        self.assertEqual(filtered[0].text, "75%→80%")

    def test_arrow_overlapping_an_ordinary_cjk_glyph_is_not_extracted(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        draw_arrow(image, (55, 24, 26, 50), "up", (0, 112, 255))
        line = SimpleNamespace(text="库存", x=20, y=24, w=61, h=50,
                               chars=[(20, 45), (45, 70)])

        filtered, arrows = detect_graphic_arrows(image, [line])

        self.assertEqual(arrows, [])
        self.assertEqual(filtered[0].text, "库存")

    def test_plus_and_exclamation_marks_are_not_arrows(self):
        image = np.full((100, 140, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (25, 45), (47, 51), (0, 112, 255), -1)
        cv2.rectangle(image, (34, 36), (38, 60), (0, 112, 255), -1)
        cv2.rectangle(image, (88, 34), (95, 56), (0, 112, 255), -1)
        cv2.circle(image, (91, 64), 3, (0, 112, 255), -1)

        _, arrows = detect_graphic_arrows(image)

        self.assertEqual(arrows, [])

    def test_long_return_arrow_is_supported(self):
        image = np.full((400, 800, 3), 255, dtype=np.uint8)
        blue = (180, 140, 90)
        cv2.line(image, (90, 160), (700, 160), blue, 8, cv2.LINE_AA)
        cv2.line(image, (700, 124), (700, 160), blue, 8, cv2.LINE_AA)
        cv2.line(image, (90, 145), (90, 160), blue, 8, cv2.LINE_AA)
        cv2.fillPoly(image, [np.asarray([(72, 148), (108, 148), (90, 120)],
                                         np.int32)], blue)

        _, arrows = detect_graphic_arrows(image)

        self.assertEqual(len(arrows), 1)
        self.assertEqual(arrows[0].direction, "right")
        self.assertGreater(arrows[0].box[2], 600)

    def test_quality_source_has_inventory_and_gray_flow_arrows(self):
        source = Path(__file__).resolve().parents[1] / "debug" / "quality_inputs" / "orig_5.png"
        image = cv2.imread(str(source))
        if image is None:
            self.skipTest("quality input image is not present")

        _, arrows = detect_graphic_arrows(image)

        self.assertTrue(any(arrow.direction == "up" and
                            abs(arrow.box[0] - 313) < 5 and
                            abs(arrow.box[1] - 672) < 5 for arrow in arrows))
        flow = [arrow for arrow in arrows if arrow.color == "gray" and
                arrow.direction == "right" and 650 < arrow.box[0] < 1300]
        self.assertGreaterEqual(len(flow), 2)


if __name__ == "__main__":
    unittest.main()
