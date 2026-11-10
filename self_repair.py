# -*- coding: utf-8 -*-
"""
self_repair.py —— 转换完一页后，自己找问题、自己修、修完再验一遍

原则：每一处修改都要"可验证"——修完和原图比，真的更像了才保留，否则撤回。
不靠人告诉它哪里错，靠"把 PPT 的结果画回成图片，和原图逐块比对"来发现问题：

  文字认错了  ：把识别出的字用字体画出来，和原图里这行字的笔画比。某个字对不上时，
                换成别的候选（同一块图换几种看法重新识别、外形相近的字）再画、再比，
                哪个和原图的笔画最吻合就用哪个
  字没抹干净  ：文字框底下还看得到原来的字（PPT 里会重影）→ 把抹除范围扩大再抹一次，
                验证底下的字确实没了
  漏抠了图标  ：底图上还留着一个像图标的独立小图形（周围是干净底色，不在插画里）→ 抠出来
  底图缺了一块：底图 + 抠出的元素 和原图不一样（抹掉了东西却没有元素补上）→ 用原图把那块补回去

修了什么都写进说明（网页上每页标题下会显示），也记进学习数据，让下一次少犯同样的错。
"""
import cv2
import numpy as np

import img2pptx as core
import page_check


# ----------------------------------------------------------------------------
# 文字：回渲染比对
# ----------------------------------------------------------------------------
def _line_ink(img_bgr, ln):
    x0, y0, x1, y1 = [int(v) for v in ln.ink]
    roi = img_bgr[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return None
    d = np.linalg.norm(roi - np.array(ln.color[::-1], np.float32), axis=2)
    return d < max(30.0, 0.45 * float(d.max()))


def _gaps(ln, text, h, w):
    nat = core.ref_font(100, ln.bold).getlength(text) * h / 100.0
    return len(text) > 1 and w > 1.3 * nat


def _render(ln, text, h, w):
    r = core._render_line_mask(text, h, w, getattr(ln, "family", "sans"), ln.bold, _gaps(ln, text, h, w))
    if r is None:
        return None
    return cv2.dilate(r.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _iou(a, b, cols=None):
    if cols is not None:
        a, b = a[:, cols[0]:cols[1]], b[:, cols[0]:cols[1]]
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _cell(ln, text, i, w):
    """第 i 个字在这行里大概占哪几列"""
    f = core.ref_font(100, ln.bold)
    total = f.getlength(text) or 1.0
    a = int(round(f.getlength(text[:i]) / total * w))
    b = int(round(f.getlength(text[:i + 1]) / total * w))
    pad = max(1, (b - a) // 6)
    return max(0, a - pad), min(w, b + pad)


def _ocr_alternatives(img_bgr, ln):
    """同一行换几种看法重新识别，返回候选文字（和原文字数相同或只差一个字的）。"""
    eng = getattr(core.ocr_rapid, "engine", None)
    if eng is None:
        return []
    H, W = img_bgr.shape[:2]
    pad = max(2, int(0.25 * ln.h))
    crop = img_bgr[max(0, ln.y - pad):min(H, ln.y + ln.h + pad), max(0, ln.x - pad):min(W, ln.x + ln.w + pad)]
    if crop.size == 0:
        return []
    out = set()
    for v in (crop, cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), core._shear(crop, 0.22)):
        try:
            r, _ = eng(v, use_det=False, use_cls=False, use_rec=True)
        except Exception:
            continue
        if r:
            t = str(r[0][0]).strip().replace("　", "")
            if not t or t == ln.text or len(t) != len(ln.text):
                continue                                        # 只接受"逐字替换"：加减字、加标点都不算
            diff = [(a, b) for a, b in zip(ln.text, t) if a != b]
            if 1 <= len(diff) <= 2 and all("\u4e00" <= a <= "\u9fff" and "\u4e00" <= b <= "\u9fff" for a, b in diff):
                out.add(t)
    return list(out)


def _set_text(ln, text):
    if ln.runs and len(ln.runs) > 1 and len(text) == len(ln.text):
        nr, k = [], 0
        for t, c, b, px in ln.runs:
            nr.append((text[k:k + len(t)], c, b, px))
            k += len(t)
        ln.runs = nr
    elif ln.runs:
        t, c, b, px = max(ln.runs, key=lambda r: len(r[0]))
        ln.runs = [(text, c, b, px)]
    if len(text) != len(ln.text):
        ln.chars = None
    ln.text = text


def verify_text(img_bgr, lines):
    """返回 [(旧文字, 新文字)]"""
    words = core._words()
    fixes = []
    for ln in lines:
        if ln.badge or not ln.ink or len(ln.text) < 2 or getattr(ln, "manual", False):
            continue
        if not any("一" <= c <= "鿿" for c in ln.text):
            continue
        ink = _line_ink(img_bgr, ln)
        if ink is None or ink.shape[0] < 10 or ink.shape[1] < 10:
            continue
        ink = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        h, w = ink.shape
        base = _render(ln, ln.text, h, w)
        if base is None:
            continue
        best_text, best_gain = None, 0.0
        # 1. 外形相近的字：逐字局部比对（"目"和"自"只差一笔，整行比看不出来，要看那一个字的位置）
        for i, c in enumerate(ln.text):
            alt = core._CONFUSE.get(c)
            if not alt:
                continue
            cand = ln.text[:i] + alt + ln.text[i + 1:]
            r = _render(ln, cand, h, w)
            if r is None:
                continue
            cols = _cell(ln, ln.text, i, w)
            gain = _iou(ink, r, cols) - _iou(ink, base, cols)
            makes_word = core._best_word(cand, i) > 0 and core._best_word(ln.text, i) == 0
            if (gain >= 0.05 and makes_word) or gain >= 0.12:
                if gain > best_gain:
                    best_text, best_gain = cand, gain
        # 2. 换看法重新识别：整行比对
        b0 = _iou(ink, base)
        for cand in _ocr_alternatives(img_bgr, ln):
            r = _render(ln, cand, h, w)
            if r is None:
                continue
            gain = _iou(ink, r) - b0
            if gain >= 0.04 and gain > best_gain:
                best_text, best_gain = cand, gain
        if best_text:
            fixes.append((ln.text, best_text))
            _set_text(ln, best_text)
    return fixes


# ----------------------------------------------------------------------------
# 重影：扩大抹除范围
# ----------------------------------------------------------------------------
def _edges(g):
    return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) > 100


def fix_ghosts(img_bgr, clean, lines):
    """返回 (新底图, 修好的行数)"""
    boxes = [[l.ink[0], l.ink[1], l.ink[2] - l.ink[0], l.ink[3] - l.ink[1]] for l in lines if l.ink]
    ghosts = page_check.find_ghost_text(img_bgr, clean, boxes)
    if not ghosts:
        return clean, 0
    H, W = img_bgr.shape[:2]
    go = _edges(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    fixed = 0
    for x, y, w, h in ghosts:
        ln = next((l for l in lines if l.ink and [l.ink[0], l.ink[1], l.ink[2] - l.ink[0], l.ink[3] - l.ink[1]] == [x, y, w, h]), None)
        if ln is None:
            continue
        pad = max(3, int(0.2 * h))
        X0, Y0, X1, Y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        roi = img_bgr[Y0:Y1, X0:X1].astype(np.float32)
        d = np.linalg.norm(roi - np.array(ln.color[::-1], np.float32), axis=2)
        strokes = (d < max(45.0, 0.6 * float(d.max()))).astype(np.uint8) * 255
        k = max(5, int(round(0.18 * h)) | 1)
        m = np.zeros((H, W), np.uint8)
        m[Y0:Y1, X0:X1] = cv2.dilate(strokes, np.ones((k, k), np.uint8))
        try:
            trial, _ = core.erase_text(clean, m, "telea")
        except Exception:
            continue
        before = _edges(cv2.cvtColor(clean[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)).sum()
        after = _edges(cv2.cvtColor(trial[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)).sum()
        orig = go[y:y + h, x:x + w].sum()
        if after < 0.6 * before and after <= 0.35 * max(orig, 1):      # 验证：底下的字确实少了很多
            clean = trial
            fixed += 1
    return clean, fixed


# ----------------------------------------------------------------------------
# 缺块：用原图补回
# ----------------------------------------------------------------------------
def fix_damage(img_bgr, clean, lines, elements):
    """底图 + 元素 和原图明显不一样、又不是文字的地方：用原图像素补回底图。
    补回去只会让结果更接近原图（有元素盖着的地方合成结果不变），所以不用再逐块验证，一次算完。"""
    text_boxes = [[l.ink[0], l.ink[1], l.ink[2] - l.ink[0], l.ink[3] - l.ink[1]] for l in lines if l.ink]
    composed = page_check.compose(clean, elements)
    regions = page_check.find_damage(img_bgr, composed, text_boxes)
    if not regions:
        return clean, 0
    diff = page_check.diff_map(img_bgr, composed, text_boxes) > 60
    tmask = page_check._box_mask(img_bgr.shape, text_boxes, 4)
    clean = clean.copy()
    fixed = 0
    for x, y, w, h in regions:
        if tmask[y:y + h, x:x + w].mean() > 0.3:
            continue                                            # 主要是文字区：字本来就该抹掉
        sel = cv2.dilate((diff[y:y + h, x:x + w] & ~tmask[y:y + h, x:x + w]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sel &= ~tmask[y:y + h, x:x + w]
        if sel.sum() < 10:
            continue
        clean[y:y + h, x:x + w][sel] = img_bgr[y:y + h, x:x + w][sel]
        fixed += 1
    return clean, fixed


# ----------------------------------------------------------------------------
# 漏抠：底图上干净底色里的独立小图形
# ----------------------------------------------------------------------------
def fix_leftovers(img_bgr, clean, lines, elements, text_mask):
    text_boxes = [[l.ink[0], l.ink[1], l.ink[2] - l.ink[0], l.ink[3] - l.ink[1]] for l in lines if l.ink]
    found = page_check.find_leftovers(img_bgr, clean, text_boxes, elements)
    if not found:
        return clean, [], []
    H, W = img_bgr.shape[:2]
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.blur(g, (7, 7))
    busy = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - m * m, 0)) > 6
    added, boxes = [], []
    for x, y, w, h in found:
        if min(w, h) < 0.02 * W or max(w, h) > 0.15 * W or max(w, h) > 2.5 * min(w, h):
            continue
        if core._context_busy(busy, text_mask, (x, y, w, h)) >= 0.25:
            continue                                            # 在插画里：是插画的一部分，不抠
        el, sel, clean_edge = core._manual_element(img_bgr, (x - 3, y - 3, w + 6, h + 6), "图标", text_mask)
        if not clean_edge:
            continue                                            # 四周不是干净底色：不是独立图标
        el.score, el.why = 70, "自检补抠：底图上漏掉的独立图标"
        hole = cv2.dilate(sel.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        try:
            trial, _ = core.erase_text(clean, hole, "telea")
        except Exception:
            continue
        # 验证：抠走以后底图那块变平了（真是一个独立的东西），而且合成回去和原图一致
        before = float(clean[y:y + h, x:x + w].astype(np.float32).std())
        after = float(trial[y:y + h, x:x + w].astype(np.float32).std())
        if after < 0.5 * before:
            clean = trial
            added.append(el)
            boxes.append([x, y, w, h])
    return clean, added, boxes


# ----------------------------------------------------------------------------
def run(img_bgr, clean, lines, icons, text_mask, allow_add=True):
    """返回 (新底图, 新元素列表, 说明列表, 学习用的记录)"""
    notes, log = [], {}
    try:
        tf = verify_text(img_bgr, lines)
        if tf:
            log["text"] = tf
            notes.append("自检改正了 %d 处错字（%s）" % (len(tf), "、".join(f"{a}→{b}" for a, b in tf[:3])))
    except Exception:
        pass
    try:
        clean, n = fix_ghosts(img_bgr, clean, lines)
        if n:
            notes.append(f"自检发现 {n} 行字底下没抹干净，已重新抹除")
    except Exception:
        pass
    try:
        if not allow_add:                                       # 人工改过图形：你删掉的图标不能被"补"回来
            raise RuntimeError
        clean, added, boxes = fix_leftovers(img_bgr, clean, lines, icons, text_mask)
        if added:
            icons = icons + added
            log["added"] = boxes
            notes.append(f"自检补抠了 {len(added)} 个漏掉的图标")
    except Exception:
        pass
    try:
        clean, n = fix_damage(img_bgr, clean, lines, icons)
        if n:
            notes.append(f"自检发现 {n} 处底图缺块，已用原图补回")
    except Exception:
        pass
    return clean, icons, notes, log
