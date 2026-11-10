import unittest

import numpy as np
from PIL import Image, ImageDraw

import img2pptx as core


def bar_chart():
    """白底上的柱状图：左边一列刻度 0–8,000，底下一条横轴，三根柱子，柱子下面是类别名。"""
    image = Image.new('RGB', (1200, 700), 'white')
    draw = ImageDraw.Draw(image)
    font = core.ref_font(20)
    lines = []
    for i, v in enumerate(range(0, 10000, 2000)):
        y = 500 - i * 80
        text = f"{v:,}"
        tw = draw.textlength(text, font=font)
        draw.text((180 - tw, y), text, fill=(60, 60, 60), font=font, anchor='lm')
        lines.append(core.Line(text, int(180 - tw), y - 12, int(tw), 24))
    draw.line((190, 500, 700, 500), fill=(170, 170, 170), width=2)     # 横轴
    for x, top, name in ((240, 300, '2024'), (400, 220, '2025'), (560, 140, '2026')):
        draw.rectangle((x, top, x + 90, 499), fill=(40, 110, 200))
        draw.text((x + 45, 530), name + '年', fill=(30, 30, 30), font=font, anchor='mm')
        lines.append(core.Line(name + '年', x + 15, 518, 60, 24))
    draw.text((900, 80), '页面大标题，不属于图表', fill=(0, 0, 0), font=font)
    lines.append(core.Line('页面大标题，不属于图表', 900, 70, 220, 26))
    return np.asarray(image)[:, :, ::-1].copy(), lines


class ChartTests(unittest.TestCase):
    def test_tick_values(self):
        self.assertEqual(core._tick_value('18,000'), 18000)
        self.assertEqual(core._tick_value('25%'), 25)
        self.assertEqual(core._tick_value('0'), 0)
        self.assertIsNone(core._tick_value('2025年'))
        self.assertIsNone(core._tick_value('+25.6%x'))
        self.assertEqual(core._tick_value('−1,000'), -1000)

    def test_bar_chart_is_one_picture(self):
        img, lines = bar_chart()
        rects = core.find_charts(img, lines)
        self.assertEqual(len(rects), 1)
        x, y, w, h = rects[0]
        for ln in lines[:-1]:                                  # 刻度和类别名都在图里
            self.assertTrue(x <= ln.x and y <= ln.y and ln.x + ln.w <= x + w and ln.y + ln.h <= y + h, ln.text)
        self.assertGreaterEqual(x + w, 650)                    # 一直到横轴的末端
        title = lines[-1]
        self.assertTrue(title.x >= x + w or title.y + title.h <= y)

    def test_table_column_is_not_a_chart(self):
        # 表格里一列右对齐的数：不是等差的刻度
        image = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(image)
        font = core.ref_font(20)
        lines = []
        for i, v in enumerate((8, 10000, 80000, 3000, 4000, 87000)):
            text = f"{v:,}"
            tw = draw.textlength(text, font=font)
            draw.text((400 - tw, 100 + i * 60), text, fill=(0, 0, 0), font=font, anchor='lm')
            lines.append(core.Line(text, int(400 - tw), 88 + i * 60, int(tw), 24))
            draw.line((100, 130 + i * 60, 700, 130 + i * 60), fill=(170, 170, 170), width=1)
        self.assertEqual(core.find_charts(np.asarray(image)[:, :, ::-1].copy(), lines), [])

    def test_risk_matrix_is_one_picture(self):
        # 左边一列"高 中 低"、底下一排"低 中 高"，中间是格子
        image = Image.new('RGB', (1000, 700), 'white')
        draw = ImageDraw.Draw(image)
        font = core.ref_font(22)
        lines = []
        for i, t in enumerate('高中低'):
            draw.text((100, 150 + i * 120), t, fill=(0, 0, 0), font=font, anchor='mm')
            lines.append(core.Line(t, 89, 139 + i * 120, 22, 22))
        for i, t in enumerate('低中高'):
            draw.text((220 + i * 140, 500), t, fill=(0, 0, 0), font=font, anchor='mm')
            lines.append(core.Line(t, 209 + i * 140, 489, 22, 22))
        draw.line((140, 90, 140, 470), fill=(40, 60, 90), width=3)
        draw.line((140, 470, 560, 470), fill=(40, 60, 90), width=3)
        for c in range(3):
            for r in range(3):
                draw.rectangle((150 + c * 140, 95 + r * 120, 285 + c * 140, 210 + r * 120), fill=(215, 228, 245))
        rects = core.find_charts(np.asarray(image)[:, :, ::-1].copy(), lines)
        self.assertEqual(len(rects), 1)
        x, y, w, h = rects[0]
        self.assertTrue(x <= 89 and y <= 90 and x + w >= 560 and y + h >= 511)

    def test_table_rows_stay_separate(self):
        image = Image.new('RGB', (600, 300), 'white')
        draw = ImageDraw.Draw(image)
        draw.line((0, 150, 600, 150), fill=(200, 205, 215), width=1)   # 表格的格线
        a = core.Line('第一格第二行', 20, 110, 200, 24, size_px=24, ink=(20, 110, 220, 134), baseline=130)
        b = core.Line('第二格第一行', 20, 160, 200, 24, size_px=24, ink=(20, 160, 220, 184), baseline=180)
        bgr = np.asarray(image)[:, :, ::-1].copy()
        self.assertEqual(len(core.group_lines([a, b])), 1)            # 不看图时按行距会并在一起
        self.assertEqual(len(core.group_lines([a, b], bgr)), 2)

    def test_axis_crossing_zero(self):
        items = [(v, 400 - v * 0.05, None) for v in (-3000, -2000, -1000, 0, 1000, 2000)]
        self.assertIsNotNone(core._tick_chain(items))

    def test_year_row_is_not_an_axis(self):
        items = [(v, 100 + i * 150, None) for i, v in enumerate((2021, 2022, 2023, 2024, 2025))]
        self.assertIsNone(core._tick_chain(items))


class IconGlyphTests(unittest.TestCase):
    def header(self, text):
        image = Image.new('RGB', (500, 60), (20, 60, 120))
        draw = ImageDraw.Draw(image)
        for dx in (10, 26):                                    # "88" 四方格图标
            for dy in (14, 30):
                draw.rectangle((dx, dy, dx + 12, dy + 12), outline='white', width=3)
        draw.text((70, 30), '供应风险矩阵', fill='white', font=core.ref_font(28), anchor='lm')
        line = core.Line(text, 8, 10, 260, 38)
        return np.asarray(image)[:, :, ::-1].copy(), line

    def test_leading_icon_is_stripped(self):
        img, line = self.header('品 供应风险矩阵')
        self.assertEqual(core.strip_icon_glyph(img, [line]), 1)
        self.assertEqual(line.text, '供应风险矩阵')
        self.assertGreater(line.x, 40)

    def test_normal_text_is_kept(self):
        img, line = self.header('品质 供应风险矩阵')
        self.assertEqual(core.strip_icon_glyph(img, [line]), 0)
        self.assertEqual(line.text, '品质 供应风险矩阵')


class WholeChartTests(unittest.TestCase):
    def donut(self, colors):
        image = Image.new('RGB', (1672, 941), (250, 248, 244))
        draw = ImageDraw.Draw(image)
        for (a, b), c in zip(((0, 120), (120, 360)), colors):
            draw.pieslice((500, 300, 730, 530), a, b, fill=c)
        draw.ellipse((550, 350, 680, 480), fill=(250, 248, 244))
        return np.asarray(image)[:, :, ::-1].copy()

    def test_segmented_ring_is_a_chart(self):
        rects = core.find_charts(self.donut([(60, 150, 110), (190, 220, 200)]), [])
        self.assertEqual(len(rects), 1)
        x, y, w, h = rects[0]
        self.assertTrue(x <= 500 and y <= 300 and x + w >= 730 and y + h >= 530)

    def test_single_colour_ring_is_not_a_chart(self):
        self.assertEqual(core.find_charts(self.donut([(60, 150, 110), (60, 150, 110)]), []), [])


class ThinPartIconTests(unittest.TestCase):
    def test_sharp_table_cell_is_not_an_icon(self):
        image = Image.new('RGB', (1672, 941), (246, 249, 252))
        ImageDraw.Draw(image).rectangle((400, 300, 600, 370), fill=(24, 55, 83))
        icons, _ = core.extract_icons(np.asarray(image)[:, :, ::-1].copy(), np.zeros((941, 1672), np.uint8))
        self.assertEqual(icons, [])

    def test_bar_chart_icon_from_thin_bars(self):
        # 四根细柱子拼成的"柱状图"图标：每根单独都太细，合起来才是一个图标
        image = Image.new('RGB', (1672, 941), (246, 249, 252))
        draw = ImageDraw.Draw(image)
        for i, hgt in enumerate((17, 26, 35, 46)):
            x = 130 + i * 14
            draw.rectangle((x, 605 - hgt, x + 9, 605), fill=(20, 94, 175))
        icons, _ = core.extract_icons(np.asarray(image)[:, :, ::-1].copy(), np.zeros((941, 1672), np.uint8))
        self.assertEqual(len(icons), 1)
        x, y, w, h = icons[0].box
        self.assertTrue(x <= 131 and x + w >= 181 and y <= 560)


if __name__ == '__main__':
    unittest.main()
