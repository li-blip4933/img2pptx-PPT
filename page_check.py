# -*- coding: utf-8 -*-
"""
page_check.py —— 转换完一页后自己检查一遍，把可疑的地方列出来

转换时的错误分两种：抠错了，合成图和原图不一样，一比就看得出来；
漏抠了，东西还留在底图里，合成图和原图一模一样，比对是发现不了的。所以三项检查一起做：
  对比：  底图 + 抠出的元素 和原图不一样的地方（元素被抹掉了、图片缺了一块）
  残留：  底图上还留着像图标的小图形（可能漏识别），文字框底下还有字（会出现重影）
  一致性：一排样子相同的东西，有的带图标、有的没带；有的字转成了文字、有的留在图里

输入都用网页里的框 [x, y, w, h, ...]。只报告，不改转换结果。
"""
import cv2
import numpy as np

import img2pptx as core


def compose(clean, elements):
    """底图 + 抠出的图片 / 图标（不含文字）"""
    out = clean.astype(np.float32)
    H, W = out.shape[:2]
    for el in sorted(elements, key=lambda e: e.kind != "图片"):
        h, w = el.rgba.shape[:2]
        x0, y0, x1, y1 = max(0, el.x), max(0, el.y), min(W, el.x + w), min(H, el.y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = el.rgba[y0 - el.y:y1 - el.y, x0 - el.x:x1 - el.x].astype(np.float32)
        a = crop[..., 3:4] / 255.0
        out[y0:y1, x0:x1] = crop[..., :3] * a + out[y0:y1, x0:x1] * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _box_mask(shape, boxes, pad=0):
    m = np.zeros(shape[:2], bool)
    H, W = shape[:2]
    for b in boxes:
        x, y, w, h = [int(v) for v in b[:4]]
        m[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)] = True
    return m


def _center_in(a, b):
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]


def diff_map(img, composed, text_boxes):
    """差异图：0–255。文字本来就被抹掉了，文字框里面不算"""
    d = np.linalg.norm(img.astype(np.float32) - composed.astype(np.float32), axis=2)
    d[_box_mask(img.shape, text_boxes, 6)] = 0
    return np.clip(d * 2, 0, 255).astype(np.uint8)


def heatmap(img, diff):
    """把差异叠在变淡的原图上：越红差得越多"""
    base = (img.astype(np.float32) * 0.35 + 255 * 0.65)
    color = cv2.applyColorMap(diff, cv2.COLORMAP_JET).astype(np.float32)
    a = (cv2.GaussianBlur(diff, (5, 5), 0).astype(np.float32) / 255.0)[..., None]
    a = np.clip(a * 3, 0, 0.9)
    return (color * a + base * (1 - a)).astype(np.uint8)


def find_damage(img, composed, text_boxes):
    """合成图和原图差得多、面积又不小的地方"""
    H, W = img.shape[:2]
    m = (diff_map(img, composed, text_boxes) > 80).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    side = max(12, int(0.012 * W))
    return [[int(v) for v in st[i][:4]] for i in range(1, n)
            if max(st[i][2], st[i][3]) >= side and min(st[i][2], st[i][3]) >= 2 and st[i][4] >= 0.25 * side * side]


def find_leftovers(img, clean, text_boxes, elements):
    """底图上还留着的、大小和样子都像图标的图形"""
    taken = (_box_mask(clean.shape, text_boxes, 3)
             | _box_mask(clean.shape, [[e.x, e.y, e.rgba.shape[1], e.rgba.shape[0]] for e in elements], 3))
    return [list(b) for b in core.find_unclaimed_shapes(img, clean, taken)]


def find_ghost_text(img, clean, lines):
    """转成文字的那几行，底图上原来的字没抹干净：PPT 里会出现两层字"""
    out = []
    gray_o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_c = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    def edges(g):
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) > 100
    eo, ec = edges(gray_o), edges(gray_c)
    for b in lines:
        x, y, w, h = [int(v) for v in b[:4]]
        if w <= 0 or h <= 0:
            continue
        before, after = eo[y:y + h, x:x + w].sum(), ec[y:y + h, x:x + w].sum()
        if before >= 30 and after > 0.35 * before:
            out.append([x, y, w, h])
    return out


def _rows(items, along, tol):
    """按一条线排开的东西分组：along=1 同一横排（y 相近），along=0 同一竖列（x 相近）"""
    groups = []
    for it in sorted(items, key=lambda b: b[along] + b[along + 2] / 2):
        c = it[along] + it[along + 2] / 2
        if groups and abs(c - groups[-1][-1][along] - groups[-1][-1][along + 2] / 2) <= tol:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def find_inconsistent(texts, icons, W):
    """一排样子相同的条目（相同大小的标签、短标题），有的配了图标、有的没配，有的字转了、有的没转"""
    out = []
    labels = [b[:4] for b in icons if b[2] >= 1.8 * b[3] and b[2] >= 0.06 * W]   # 扁长的标签底板
    shorts = [b[:4] for b in texts if b[3] <= 0.05 * W and len(str(b[5])) <= 12 and "\n" not in str(b[5])]
    plain = [b[:4] for b in icons]                             # 图标被拆成扁扁的两半时也算有图标

    def even(g, along):
        ctr = sorted(b[1 - along] + b[3 - along] / 2 for b in g)
        gaps = np.diff(ctr)
        return len(gaps) >= 2 and gaps.min() > 0 and gaps.max() <= 1.5 * gaps.min()

    def has_icon(b, side):
        for p in plain:
            if list(p) == list(b):
                continue
            if side == "above":
                d = b[1] - (p[1] + p[3])                       # 图标和条目之间可能还隔着一行小标题
                if -0.3 * p[3] <= d <= 0.1 * W and p[0] < b[0] + b[2] and p[0] + p[2] > b[0]:
                    return True
            else:
                d = b[0] - (p[0] + p[2])
                if -0.3 * p[2] <= d <= 2.5 * max(b[3], p[3]) and p[1] < b[1] + b[3] and p[1] + p[3] > b[1]:
                    return True
        return False

    # 同样大小的标签：字有的转了有的没转
    for g in _rows(labels, 1, 0.012 * W):
        same = [b for b in g if abs(b[2] - g[0][2]) <= 0.08 * g[0][2] and abs(b[3] - g[0][3]) <= 0.12 * g[0][3]]
        if len(same) < 3:
            continue
        flags = [any(_center_in(t[:4], b) for t in texts) for b in same]
        if 0 < sum(flags) < len(flags):
            bad = [b for b, f in zip(same, flags) if f != (2 * sum(flags) >= len(flags))]
            out += [(list(b), "同一排标签里，这个的文字和其他几个处理得不一样") for b in bad]
    # 一排条目：有的配图标、有的没配
    for along, side in ((1, "above"), (0, "left")):
        for g in _rows(labels + shorts, along, 0.012 * W):
            g = [b for b in g if abs(b[3] - g[0][3]) <= 0.3 * g[0][3]]
            if len(g) < 3 or not even(g, along):
                continue
            flags = [has_icon(b, side) for b in g]
            n = sum(flags)
            if n >= 2 and n >= 0.6 * len(flags) and n < len(flags):
                out += [(list(b), "同一排其他条目都配了图标，这一个没有识别到") for b, f in zip(g, flags) if not f]
    uniq, seen = [], set()
    for b, msg in out:
        if tuple(b) not in seen:
            seen.add(tuple(b))
            uniq.append((b, msg))
    return uniq


def check_page(img, clean, texts, icons, pictures, lines, elements):
    """返回 (问题列表 [[x, y, w, h, 说明], ...], 差异热力图)。三项检查各自出错都不影响转换结果"""
    H, W = img.shape[:2]
    composed = compose(clean, elements)
    text_boxes = [b[:4] for b in texts]
    issues = []
    for b in find_damage(img, composed, text_boxes):
        issues.append(b + ["和原图不一样：这里可能被抹掉了内容或缺了一块"])
    for b in find_leftovers(img, clean, text_boxes, elements):
        issues.append(b + ["底图上还有一个像图标的图形，可能漏识别了"])
    for b in find_ghost_text(img, clean, lines):
        issues.append(b + ["这行字底下原来的字没抹干净，可能出现重影"])
    for b, msg in find_inconsistent(texts, icons, W):
        issues.append([int(v) for v in b] + [msg])
    return issues, heatmap(img, diff_map(img, composed, text_boxes))
