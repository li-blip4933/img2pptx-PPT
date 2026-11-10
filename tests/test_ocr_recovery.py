import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import img2pptx


def item(text, x, y, w, h, score=0.96, chars=None):
    quad = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    if chars is None:
        return [quad, text, score]
    char_boxes = []
    for x0, x1 in chars:
        char_boxes.append([[x0, y], [x1, y], [x1, y + h], [x0, y + h]])
    return [quad, text, score, char_boxes, list(text)]


class FakeEngine:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image.shape[:2], kwargs))
        return next(self.responses), None


class OCRRecoveryTests(unittest.TestCase):
    def run_fake(self, responses, shape=(1000, 1700, 3)):
        engine = FakeEngine(responses)
        fake_module = types.SimpleNamespace(
            RapidOCR=lambda **kwargs: self.capture_engine(engine, kwargs))
        img2pptx.ocr_rapid.__dict__.pop("engine", None)
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": fake_module}):
            lines = img2pptx.ocr_rapid(np.zeros(shape, dtype=np.uint8))
        return lines, engine

    @staticmethod
    def capture_engine(engine, kwargs):
        engine.init_kwargs = kwargs
        return engine

    def test_tile_offsets_restore_line_and_character_coordinates(self):
        main = [item("整页", 50, 60, 80, 30, chars=[(50, 90), (90, 130)])]
        tiles = [
            [],
            [item("补充", 300, 200, 80, 30, chars=[(300, 340), (340, 380)])],
            [], [],
        ]
        lines, engine = self.run_fake([main] + tiles)
        self.assertEqual(engine.init_kwargs, {"intra_op_num_threads": 4,
                                               "inter_op_num_threads": 1})
        supplement = next(line for line in lines if line.text == "补充")
        self.assertEqual((supplement.x, supplement.y, supplement.w, supplement.h),
                         (1000, 200, 80, 30))
        self.assertEqual(supplement.chars, [(1000, 1040), (1040, 1080)])
        self.assertEqual(lines[0].chars, [(50, 90), (90, 130)])

    def test_overlapping_tiles_deduplicate_without_merging_columns(self):
        main = []
        # Tile sequence for 1700x1000: origins (0,0), (700,0), (0,300), (700,300).
        tiles = [
            [item("左栏", 300, 200, 50, 28), item("重叠", 850, 200, 70, 28)],
            [item("重叠", 150, 200, 70, 28), item("右栏", 380, 200, 50, 28)],
            [], [],
        ]
        lines, _ = self.run_fake([main] + tiles)
        text = [line.text for line in lines]
        self.assertEqual(text.count("重叠"), 1)
        self.assertIn("左栏", text)
        self.assertIn("右栏", text)
        self.assertEqual((next(line for line in lines if line.text == "右栏").x), 1080)

    def test_tile_edge_fragment_is_ignored_and_main_line_is_preserved(self):
        main = [item("整行保留", 12, 160, 180, 30)]
        tiles = [
            [], [item("截断候选", 2, 300, 60, 28)], [], [],
        ]
        lines, _ = self.run_fake([main] + tiles)
        self.assertEqual([line.text for line in lines], ["整行保留"])

    def test_small_image_does_not_trigger_tiles(self):
        engine = FakeEngine([[item("小图", 10, 10, 30, 20)]])
        fake_module = types.SimpleNamespace(RapidOCR=lambda **_: engine)
        img2pptx.ocr_rapid.__dict__.pop("engine", None)
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": fake_module}):
            lines = img2pptx.ocr_rapid(np.zeros((900, 1400, 3), dtype=np.uint8))
        self.assertEqual([line.text for line in lines], ["小图"])
        self.assertEqual(len(engine.calls), 1)

    def test_confident_single_character_label_is_not_discarded(self):
        label = img2pptx.Line("高", 10, 10, 30, 30, score=0.98)
        uncertain = img2pptx.Line("高", 10, 10, 30, 30, score=0.75)
        self.assertEqual(img2pptx._clean_lines([label], 500), [label])
        self.assertEqual(img2pptx._clean_lines([uncertain], 500), [])

    def test_two_supported_rows_replace_mixed_vertical_detection(self):
        line = img2pptx.Line("目P", 100, 80, 40, 100, score=0.95)
        responses = [
            [],
            [item("P", 22, 5, 12, 22, score=0.95)],
            [item("P", 22, 5, 12, 22, score=0.96),
             item("计划", 42, 12, 30, 24, score=0.97)],
            [item("计划", 42, 12, 30, 24, score=0.98)],
        ]
        groups = img2pptx._recover_tall_line(FakeEngine(responses),
                                              np.zeros((250, 250, 3), dtype=np.uint8), line)
        result = img2pptx._recoveries_for_mixed_line(groups, line)
        self.assertEqual(set(groups), {"P", "计划"})
        self.assertEqual({ln.text for ln in result}, {"P", "计划"})
        self.assertTrue(all(ln.score >= 0.85 for ln in result))

    def test_stable_ascii_substring_can_replace_mixed_line_alone(self):
        line = img2pptx.Line("目P", 100, 80, 40, 100, score=0.95)
        p = img2pptx.Line("P", 100, 110, 12, 22, score=0.96)
        groups = {"P": {"bands": {1, 2}, "best": p}}
        self.assertEqual(img2pptx._recoveries_for_mixed_line(groups, line), [p])

    def test_overlapping_known_long_text_is_preserved_over_short_strip_result(self):
        mixed = img2pptx.Line("Qc", 100, 80, 40, 100, score=0.95)
        known = img2pptx.Line("检查", 100, 132, 42, 30, score=0.99)
        responses = [
            [],
            [item("C", 22, 28, 14, 20, score=0.96)],
            [item("检查", 22, 30, 42, 30, score=0.99)],
            [],
        ]
        result = img2pptx._supplement_or_replace_mixed(
            FakeEngine(responses), np.zeros((250, 250, 3), dtype=np.uint8),
            mixed, [mixed, known])
        self.assertEqual([line.text for line in result], ["C"])

    def test_single_ascii_main_glyph_is_kept_with_repeated_chinese_row(self):
        line = img2pptx.Line("A", 100, 80, 40, 100, score=0.95)
        rows = [item("改进", 24, 10, 50, 24, score=0.96)]
        engine = FakeEngine([[], rows, rows, []])
        result = img2pptx._supplement_or_replace_mixed(
            engine, np.zeros((250, 250, 3), dtype=np.uint8), line, [line])
        self.assertEqual([ln.text for ln in result], ["A", "改进"])

    def test_low_confidence_tall_single_cjk_icon_is_removed_without_strip_support(self):
        line = img2pptx.Line("目", 50, 30, 52, 68, score=0.80)
        engine = FakeEngine([[], []])
        kept = img2pptx._filter_main_icon_glyphs(
            engine, np.zeros((200, 200, 3), dtype=np.uint8), [line])
        self.assertEqual(kept, [])

    def test_tile_single_cjk_character_is_not_supplemented(self):
        engine = FakeEngine([[], [item("二", 300, 200, 24, 30)], [], [], []])
        fake_module = types.SimpleNamespace(RapidOCR=lambda **_: engine)
        img2pptx.ocr_rapid.__dict__.pop("engine", None)
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": fake_module}):
            lines = img2pptx.ocr_rapid(np.zeros((1000, 1700, 3), dtype=np.uint8))
        self.assertNotIn("二", [line.text for line in lines])
        self.assertEqual(len(engine.calls), 5)


if __name__ == "__main__":
    unittest.main()
