import unittest

import cv2
import numpy as np

import img2pptx as core
import page_check as pc


def page(W=1600, H=900):
    return np.full((H, W, 3), (245, 248, 250), np.uint8)


def icon_at(x, y, size=50):
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = (120, 60, 20)
    rgba[..., 3] = 255
    return core.Icon(x, y, rgba, "图标", (x, y, size, size), 100, "")


class CompareTests(unittest.TestCase):
    def test_extracted_icon_matches_source(self):
        img, clean = page(), page()
        img[100:150, 100:150] = (120, 60, 20)
        self.assertEqual(pc.find_damage(img, pc.compose(clean, [icon_at(100, 100)]), []), [])

    def test_erased_content_is_reported(self):
        img, clean = page(), page()
        img[100:150, 100:150] = (120, 60, 20)
        found = pc.find_damage(img, pc.compose(clean, []), [])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][:2], [100, 100])

    def test_text_area_does_not_count(self):
        img, clean = page(), page()
        img[100:150, 100:300] = (40, 40, 40)
        self.assertEqual(pc.find_damage(img, clean, [[100, 100, 200, 50]]), [])


class LeftoverTests(unittest.TestCase):
    def test_icon_left_in_background_is_reported(self):
        # 没抠出来的图标：合成图和原图完全一样，只有这项查得出来
        img = page()
        cv2.circle(img, (300, 300), 26, (120, 60, 20), 5)
        found = pc.find_leftovers(img, img.copy(), [], [])
        self.assertEqual(len(found), 1)

    def test_extracted_and_thin_lines_are_ignored(self):
        img = page()
        img[300:303, 100:900] = (120, 60, 20)                  # 分隔线
        cv2.circle(img, (300, 500), 26, (120, 60, 20), 5)      # 已经抠走的图标
        self.assertEqual(pc.find_leftovers(img, img.copy(), [], [icon_at(270, 470, 60)]), [])


class GhostTextTests(unittest.TestCase):
    def test_text_not_erased_is_reported(self):
        img = page()
        cv2.putText(img, 'REPORT', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
        box = [90, 150, 300, 70]
        self.assertEqual(pc.find_ghost_text(img, img.copy(), [box]), [box])
        self.assertEqual(pc.find_ghost_text(img, page(), [box]), [])


class ConsistencyTests(unittest.TestCase):
    W = 1672

    def test_labels_handled_differently(self):
        # 四个一样的标签，只有两个的字转成了文字
        labels = [[66 + i * 207, 629, 177, 54] for i in range(4)]
        texts = [[98, 642, 116, 28, 100, '供需分析'], [307, 642, 114, 28, 100, '竞争格局']]
        found = pc.find_inconsistent(texts, labels, self.W)
        self.assertEqual(sorted(b[0] for b, _ in found), [480, 687])

    def test_row_with_one_missing_icon(self):
        labels = [[66 + i * 207, 629, 177, 54] for i in range(4)]
        icons = labels + [[110 + i * 207, 553, 60, 60] for i in range(1, 4)]
        texts = [[b[0] + 30, 642, 116, 28, 100, '标签'] for b in labels]
        found = pc.find_inconsistent(texts, icons, self.W)
        self.assertEqual({b[0] for b, _ in found}, {66})

    def test_consistent_row_is_quiet(self):
        labels = [[66 + i * 207, 629, 177, 54] for i in range(4)]
        icons = labels + [[110 + i * 207, 553, 60, 60] for i in range(4)]
        texts = [[b[0] + 30, 642, 116, 28, 100, '标签'] for b in labels]
        self.assertEqual(pc.find_inconsistent(texts, icons, self.W), [])


if __name__ == '__main__':
    unittest.main()
