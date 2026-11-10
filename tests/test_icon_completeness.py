import unittest

import cv2
import numpy as np

import img2pptx as core


class IconCompletenessTests(unittest.TestCase):
    @staticmethod
    def _extract(image, text_mask=None):
        if text_mask is None:
            text_mask = np.zeros(image.shape[:2], np.uint8)
        icons, _ = core.extract_icons(image, text_mask)
        return icons

    def test_pale_closed_report_outline_keeps_internal_number_in_image(self):
        image = np.full((675, 1200, 3), 255, np.uint8)
        # A sparse, pale document symbol encloses text that should stay rasterized.
        cv2.rectangle(image, (200, 165), (315, 295), (218, 229, 240), 3, cv2.LINE_AA)
        cv2.line(image, (220, 190), (292, 190), (218, 229, 240), 2, cv2.LINE_AA)
        cv2.line(image, (220, 207), (292, 207), (218, 229, 240), 2, cv2.LINE_AA)
        cv2.line(image, (220, 224), (275, 224), (218, 229, 240), 2, cv2.LINE_AA)
        cv2.putText(image, "90", (228, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (28, 36, 54), 3, cv2.LINE_AA)
        cv2.line(image, (228, 283), (282, 283), (28, 104, 226), 3, cv2.LINE_AA)

        text_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.putText(text_mask, "90", (228, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    255, 3, cv2.LINE_AA)
        icons = self._extract(image, text_mask)

        report = [ic for ic in icons if self._overlap(ic.box, (196, 161, 124, 138)) > 0.7]
        self.assertEqual(len(report), 1, "the pale report outline should be one extracted icon")
        icon = report[0]
        alpha = icon.rgba[..., 3]
        local_text = text_mask[icon.y:icon.y + alpha.shape[0], icon.x:icon.x + alpha.shape[1]] > 0
        glyphs = local_text & (alpha >= 128)
        self.assertGreater(int(local_text.sum()), 0)
        self.assertGreaterEqual(glyphs.sum() / local_text.sum(), 0.98,
                                "the enclosed 90 must be opaque inside the icon")
        source_crop = image[icon.y:icon.y + alpha.shape[0], icon.x:icon.x + alpha.shape[1]]
        np.testing.assert_array_equal(icon.rgba[..., :3][glyphs], source_crop[glyphs])
        self.assertGreater(int(alpha[alpha.shape[0] // 2, alpha.shape[1] // 2]), 128,
                           "the closed outline must have opaque interior, not only a transparent ring")

    def test_disconnected_pale_bars_join_the_multicolour_curved_arrow_graph(self):
        image = np.full((788, 1400, 3), 255, np.uint8)
        base = 400
        bars = [
            (700, 716, 22, (225, 235, 247)),
            (728, 744, 40, (180, 205, 233)),
            (784, 800, 105, (20, 45, 105)),
        ]
        for x0, x1, height, color in bars:
            cv2.rectangle(image, (x0, base - height), (x1, base), color, -1, cv2.LINE_AA)
        curve = np.array([[695, 330], [725, 322], [750, 310], [780, 295], [820, 240]], np.int32)
        cv2.polylines(image, [curve], False, (35, 120, 235), 8, cv2.LINE_AA)
        cv2.fillPoly(image, [np.array([[820, 240], [801, 251], [812, 261]], np.int32)],
                     (35, 120, 235), cv2.LINE_AA)

        icons = self._extract(image)
        graph = [ic for ic in icons if self._overlap(ic.box, (690, 235, 136, 170)) > 0.5]
        self.assertEqual(len(graph), 1, "bars and curve should be one coherent extracted icon")
        icon = graph[0]
        alpha = icon.rgba[..., 3]
        source_crop = image[icon.y:icon.y + alpha.shape[0], icon.x:icon.x + alpha.shape[1]]
        expected = np.max(np.abs(source_crop.astype(np.int16) - 255), axis=2) > 35
        self.assertGreater(int(expected.sum()), 100)
        covered = (alpha >= 128) & expected
        self.assertGreaterEqual(covered.sum() / expected.sum(), 0.98,
                                "all four bars and the arrow should be retained without alpha holes")
        np.testing.assert_array_equal(icon.rgba[..., :3][covered], source_crop[covered])
        left_bar = (source_crop[..., 0] < 250) & (source_crop[..., 1] < 250) & (source_crop[..., 2] < 250)
        self.assertGreater(int(((alpha >= 128) & left_bar).sum()), 100,
                           "the pale left bar must be part of the same opaque source-RGBA image")

    def test_pale_connector_does_not_further_expand_a_filled_circle_icon(self):
        image = np.full((675, 1200, 3), 255, np.uint8)
        center, radius = (800, 360), 55
        cv2.circle(image, center, radius, (105, 45, 10), -1, cv2.LINE_AA)
        cv2.rectangle(image, (786, 327), (814, 349), (245, 248, 250), 2, cv2.LINE_AA)
        cv2.line(image, (791, 334), (809, 334), (245, 248, 250), 2, cv2.LINE_AA)
        cv2.line(image, (791, 340), (809, 340), (245, 248, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "P", (790, 392), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (245, 248, 250), 2, cv2.LINE_AA)
        # This pale arrowhead touches the circle rim but belongs to the connector.
        cv2.fillPoly(image, [np.array([[853, 350], [873, 360], [853, 370]], np.int32)],
                     (225, 235, 245), cv2.LINE_AA)
        text_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.putText(text_mask, "P", (790, 392), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)

        icons = self._extract(image, text_mask)
        circle = [ic for ic in icons if self._overlap(ic.box, (744, 304, 112, 112)) > 0.8]
        self.assertEqual(len(circle), 1, "the filled circle should remain a single icon")
        icon = circle[0]
        self.assertEqual(icon.box, (744, 304, 130, 113),
                         "keep the pre-merge bounds; the legacy pale grow already adds a faint connector tail")
        local_alpha = icon.rgba[center[1] - icon.y, center[0] - icon.x, 3]
        self.assertGreaterEqual(int(local_alpha), 128, "the circle interior remains opaque")
        connector_tip = (873, 360)
        self.assertTrue(icon.x <= connector_tip[0] < icon.x + icon.rgba.shape[1])
        self.assertLess(int(icon.rgba[connector_tip[1] - icon.y,
                                     connector_tip[0] - icon.x, 3]), 128,
                        "the faint connector tail stays below the opaque circle support")

    @staticmethod
    def _overlap(box, target):
        x, y, w, h = box
        tx, ty, tw, th = target
        intersection = max(0, min(x + w, tx + tw) - max(x, tx)) * max(0, min(y + h, ty + th) - max(y, ty))
        return intersection / float(max(1, tw * th))


class ArrowIntegrationTests(unittest.TestCase):
    def test_arrow_dedup_preserves_attached_non_arrow_part(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        image = np.full((788, 1400, 3), 255, np.uint8)
        polygon = np.array([(500, 312), (550, 312), (550, 302), (580, 320),
                            (550, 338), (550, 328), (500, 328)], np.int32)
        cv2.fillPoly(image, [polygon], (0, 120, 245))
        cv2.circle(image, (591, 320), 8, (170, 70, 10), -1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / 'combined.png'
            cv2.imwrite(str(path), image)
            _, _, icons, _, _ = core.process_image(str(path), 'rapid', 'eng',
                                                  ocr=[], with_pictures=False, inpaint='telea')
        preserved = any(icon.x <= 591 < icon.x + icon.rgba.shape[1]
                        and icon.y <= 320 < icon.y + icon.rgba.shape[0]
                        and icon.rgba[320 - icon.y, 591 - icon.x, 3] >= 128
                        and np.array_equal(icon.rgba[320 - icon.y, 591 - icon.x, :3], image[320, 591])
                        for icon in icons)
        self.assertTrue(preserved, 'arrow dedup must retain the blue component')


if __name__ == "__main__":
    unittest.main()
