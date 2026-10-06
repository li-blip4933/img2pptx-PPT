#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
img2pptx.py —— 把幻灯片图片转成"文字可编辑"的 PPTX

流程：
  1. OCR      识别每行文字的内容和位置
  2. 分析      估计每行的文字颜色、字号、是否加粗
  3. 抠图      图标、照片、插画各自抠成透明背景的独立图片
  4. 抹除      把文字和抠走的东西从原图上抹掉，并补全背景
  5. 重建      抹干净的图做底图，叠上独立图片和真正的文本框

用法：
  python img2pptx.py slide1.png slide2.png -o out.pptx
  python img2pptx.py 图片文件夹 -o out.pptx
  python img2pptx.py slide.png -o out.pptx --debug     # 额外输出中间结果图

依赖：
  pip install opencv-python pillow numpy python-pptx
  OCR 二选一：
    pip install rapidocr-onnxruntime      （推荐，中英文都准）
    或安装 tesseract 命令行                （备用，中文需 chi_sim 语言包）
"""
import argparse
import copy
import csv
import io
import os
import subprocess
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

SLIDE_W_EMU = 12192000          # 13.333 英寸，标准 16:9 宽度
EMU_PER_PT = 12700
LINE_SPACING = 1.2              # 文本框行距 = 字号 × 1.2（固定值，便于精确定位）
BASELINE_RATIO = 0.8            # 固定行距下，基线大约在行顶往下 0.8 × 行距处
BOLD_MIN_PX = 24                # 字高小于这个像素数时不判断粗细
ICON_MIN_RATIO = 0.011          # 图标最小边长 = 图宽 × 这个比例（按最小单元找，小箭头也算）
ICON_KEEP_SCORE = 55            # 图标置信分达到这个数才抠出来
ICON_MAX_RATIO = 0.12           # 图标最大边长 = 图宽 × 这个比例（更大的当作背景装饰）
PIC_MIN_RATIO = 0.045           # 图片（照片 / 插画）最小边长 = 图宽 × 这个比例
ICON_RING_CLEAN = 0.65          # 图标四周至少这么大比例的像素是同一种背景色
TEXT_RING_CLEAN = 0.55          # 小字四周同色像素比例低于此值，视为画面里的字，不转换
MIN_OCR_SCORE = 0.7             # OCR 置信度低于此值的结果丢弃（多半是图标被误认成字）


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class Line:
    text: str
    x: int
    y: int
    w: int
    h: int
    color: tuple = (0, 0, 0)
    size_px: float = 0.0        # 字号（像素）
    bold: bool = False
    baseline: float = 0.0       # 基线的 y 坐标（像素）
    ink: tuple = None           # 实际笔画的包围盒 (x0, y0, x1, y1)
    score: float = 1.0          # OCR 置信度 0–1
    badge: bool = False         # 这行字是不是某个序号圆圈里的字
    disc: tuple = None          # 所在圆圈的 (圆心 x, 圆心 y, 半径)
    italic: bool = False        # 是不是斜体
    slant: tuple = (0.0, 1.0, 0.0)   # 量出来的倾斜程度 (斜率, 比不倾斜时清晰多少倍)
    fit: float = 0.0            # 字距修正（相对字号的比例）：原图的字比参考字体窄 / 宽多少
    chars: list = None          # 每个字的横向范围 [(x0, x1)]，OCR 能提供时才有
    runs: list = None           # 一行内按颜色拆开的片段 [(文字, 颜色, 是否加粗, 字号像素)]


@dataclass
class Block:
    lines: list = field(default_factory=list)
    align: str = "left"


# ----------------------------------------------------------------------------
# 参考字体：只用来"量尺寸"，不是最终 PPT 里的字体
# ----------------------------------------------------------------------------
_FONT_CANDIDATES = {
    False: [
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),       # macOS：冬青黑体 W3
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    True: [
        "C:/Windows/Fonts/msyhbd.ttc",
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),       # macOS：冬青黑体 W6
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}
_font_cache = {}


def ref_font(size, bold=False):
    key = (int(size), bold)
    if key not in _font_cache:
        for cand in _FONT_CANDIDATES[bold]:
            path, index = cand if isinstance(cand, tuple) else (cand, 0)
            if os.path.exists(path):
                _font_cache[key] = ImageFont.truetype(path, int(size), index=index)
                break
        else:
            raise RuntimeError("找不到参考字体，请在 _FONT_CANDIDATES 里加上你电脑的字体路径")
    return _font_cache[key]


def default_ppt_font():
    return "Microsoft YaHei"        # Windows 自带；Mac 版 PowerPoint 也自带，跨电脑打开最稳


# ----------------------------------------------------------------------------
# 第 1 步：OCR
# ----------------------------------------------------------------------------
def ocr_rapid(img_bgr):
    from rapidocr_onnxruntime import RapidOCR
    if not hasattr(ocr_rapid, "engine"):
        ocr_rapid.engine = RapidOCR()
    try:                                                       # 新版本能给出每个字的位置，拆分多色文字时更准
        result, _ = ocr_rapid.engine(img_bgr, return_word_box=True)
    except TypeError:
        result, _ = ocr_rapid.engine(img_bgr)
    lines = []
    for item in result or []:
        quad, text, score = item[0], item[1], item[2]
        if float(score) < MIN_OCR_SCORE or not text.strip():
            continue
        q = np.array(quad)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        ln = Line(text.strip(), int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        ln.score = float(score)
        if len(item) >= 5 and "".join(item[4]) == ln.text and all(len(c) == 1 for c in item[4]):
            ln.chars = [(int(np.array(bx)[:, 0].min()), int(np.array(bx)[:, 0].max())) for bx in item[3]]
        lines.append(ln)
    return lines


def ocr_tesseract(img_bgr, lang):
    """备用方案：调用 tesseract 命令行，用它自带的"行"分组，再按大间距拆栏。"""
    H, W = img_bgr.shape[:2]
    scale = 2 if W < 2400 else 1            # 小图放大后识别更准
    big = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", big)
    out = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", lang, "--psm", "11", "tsv"],
        input=buf.tobytes(), capture_output=True, check=True).stdout.decode("utf-8", "ignore")
    rows = {}
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        if not t or float(r["conf"]) < 50:
            continue
        x, y, w, h = (int(int(r[k]) / scale) for k in ("left", "top", "width", "height"))
        rows.setdefault((r["block_num"], r["par_num"], r["line_num"]), []).append((x, y, w, h, t))
    lines = []
    for row in rows.values():
        med_h = float(np.median([b[3] for b in row]))
        # 去掉被误认成字符的图形（项目符号、竖条等）：单个符号，或高度明显离群
        row = [b for b in row if not (len(b[4]) == 1 and not b[4].isalnum()) and b[3] < 1.8 * med_h]
        if not row:
            continue
        row.sort(key=lambda b: b[0])
        seg = [row[0]]
        for wd in row[1:]:
            prev = seg[-1]
            hh = max(b[3] for b in seg)
            if wd[0] - (prev[0] + prev[2]) > 1.5 * hh:      # 间距太大，说明是另一栏
                lines.append(_join(seg, lang))
                seg = [wd]
            else:
                seg.append(wd)
        lines.append(_join(seg, lang))
    return [l for l in lines if len(l.text) > 1 or l.text.isalnum()]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def _join(seg, lang):
    x0 = min(b[0] for b in seg)
    y0 = min(b[1] for b in seg)
    x1 = max(b[0] + b[2] for b in seg)
    y1 = max(b[1] + b[3] for b in seg)
    # 中文词之间不加空格；两个英文 / 数字单词之间要加（中英混排的页面两种情况都有）
    text = ""
    for b in seg:
        if text and text[-1].isascii() and b[4][:1].isascii() and not text[-1].isspace():
            text += " "
        text += b[4]
    return Line(text, x0, y0, x1 - x0, y1 - y0)


def run_ocr(img_bgr, engine, lang):
    if engine in ("auto", "rapid"):
        try:
            return ocr_rapid(img_bgr), "rapidocr"
        except ImportError:
            if engine == "rapid":
                raise
    return ocr_tesseract(img_bgr, lang), "tesseract"


# ----------------------------------------------------------------------------
# 第 2 步：分析每行文字的颜色 / 字号 / 粗细，同时得到"笔画掩膜"
# ----------------------------------------------------------------------------
def analyze_line(img_bgr, ln, full_mask):
    H, W = img_bgr.shape[:2]
    pad = max(3, int(ln.h * 0.25))
    x0, y0 = max(0, ln.x - pad), max(0, ln.y - pad)
    x1, y1 = min(W, ln.x + ln.w + pad), min(H, ln.y + ln.h + pad)
    roi = img_bgr[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return False

    # 背景色：把 OCR 框内的像素聚成两类，占多数的一类是背景（深底白字也适用）
    core_px = img_bgr[ln.y:ln.y + ln.h, ln.x:ln.x + ln.w].reshape(-1, 3).astype(np.float32)
    if len(core_px) < 8:
        return False
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(core_px, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    major = int(np.bincount(labels.ravel(), minlength=2).argmax())
    # 文字很少碰到"比 OCR 框大一圈"的那条边线，所以边线上的颜色基本就是背景色；
    # 粗体大标题的笔画可能占到框内一半以上，光看"谁占多数"会把字和底认反
    ex0, ey0, ex1, ey1 = max(0, ln.x - 2), max(0, ln.y - 2), min(W, ln.x + ln.w + 2), min(H, ln.y + ln.h + 2)
    edge = np.concatenate([img_bgr[ey0, ex0:ex1], img_bgr[ey1 - 1, ex0:ex1],
                           img_bgr[ey0:ey1, ex0], img_bgr[ey0:ey1, ex1 - 1]]).astype(np.float32)
    d_edge = np.linalg.norm(centers - np.median(edge, axis=0), axis=1)
    bg = centers[int(d_edge.argmin())] if d_edge.min() < 60 else centers[major]
    # 边线上的颜色和两类都对不上（文字印在一条渐变色带上，色带边缘是过渡色）时：
    # 看文字框最里面一圈像素归哪一类 —— 笔画很少贴着文字框的边，贴边的那一类是背景
    if d_edge.min() >= 60 and ln.h >= 6 and ln.w >= 6:
        lab2d = labels.reshape(ln.h if ln.y + ln.h <= H else H - ln.y, -1)
        frame = np.ones(lab2d.shape, bool)
        frame[2:-2, 2:-2] = False
        share1 = float((lab2d[frame] == 1).mean())
        if abs(share1 - 0.5) >= 0.1:
            bg = centers[1 if share1 > 0.5 else 0]

    # 离背景色越远，越可能是文字笔画；用 Otsu 自动找分界
    dist = np.linalg.norm(roi - bg, axis=2)
    d8 = np.clip(dist, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr * 0.6, 22)
    ink = dist > thr
    # 只保留和 OCR 框相交的连通块，避免把隔壁行、边框线算进来
    n, lab = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    cx0, cx1 = ln.x, ln.x + ln.w
    if ln.chars:                                               # OCR 的框常常比字宽，会把紧挨着的图标也圈进来；
        cx0 = max(cx0, ln.chars[0][0] - 2)                     # 左边以第一个字为准（图标一般在文字左边）
        cx1 = max(cx1, ln.chars[-1][1] + 2)                    # 右边不收：句末的标点常常没有逐字位置，收了会漏掉
    core = lab[ln.y - y0:ln.y - y0 + ln.h, max(0, cx0 - x0):max(1, cx1 - x0)]
    keep = set(np.unique(core[core > 0]).tolist())
    # 文字印在一块小底板（色条、标签）上时，底板外面的背景也"和底板颜色不同"，会被当成笔画。
    # 区分办法：真正的笔画不会一直延伸到取样范围的最外圈（外圈比文字框大出四分之一个字高）
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    if 0 < y0 and y1 < H and 0 < x0 and x1 < W:
        inner_keep = keep - edge_ids
        if inner_keep and np.isin(lab, list(inner_keep)).sum() >= 0.25 * np.isin(lab, list(keep)).sum():
            keep = inner_keep
    ink = np.isin(lab, list(keep))
    # 再按颜色筛一遍：笔画的颜色应该靠近"文字色"。离背景色很远、但离文字色也很远的像素
    # （比如绿底黄字旁边漏进来的白色描边、色带外面的天空）不是这行字的笔画
    text_c = centers[1] if np.linalg.norm(centers[0] - bg) < np.linalg.norm(centers[1] - bg) else centers[0]
    contrast = float(np.linalg.norm(text_c - bg))
    if contrast > 60:
        near_text = np.linalg.norm(roi - text_c, axis=2) < 0.62 * contrast
        cy0, cy1, cxa, cxb = ln.y - y0, ln.y - y0 + ln.h, ln.x - x0, ln.x - x0 + ln.w
        before = int(ink[cy0:cy1, cxa:cxb].sum())
        after = int((ink & near_text)[cy0:cy1, cxa:cxb].sum())
        if before and after >= 0.5 * before and (ink & ~near_text).sum() >= 0.08 * ink.sum():
            ink &= near_text
    if ink.sum() < 8:
        return False

    # 文字颜色：取最"纯"的那部分笔画像素（避开抗锯齿的边缘）
    strong = ink & (dist >= np.percentile(dist[ink], 60))      # 只在笔画里取（旁边更亮的背景不能算进来）
    b, g, r = np.median(roi[strong], axis=0)
    ln.color = (int(r), int(g), int(b))

    ys, xs = np.where(ink)
    ln.ink = (int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max() + 1), int(y0 + ys.max() + 1))
    ink_w, ink_h = ln.ink[2] - ln.ink[0], ln.ink[3] - ln.ink[1]
    ln.slant = _slant(ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1])

    # 字号：用参考字体把同样的文字渲染一遍，比较笔画高度和宽度
    f = ref_font(200)
    bx0, by0, bx1, by1 = f.getbbox(ln.text, anchor="ls")     # 相对基线的包围盒
    if bx1 - bx0 <= 0 or by1 - by0 <= 0:
        return False
    s_h = ink_h / (by1 - by0)
    s_w = ink_w / (bx1 - bx0)
    scale = s_h if s_h <= s_w * 1.25 else s_w * 1.1          # 以高度为准，太宽时让步给宽度
    if ln.badge:
        scale = s_h                                           # 序号只有一个字，字宽因字体差别很大（比如"1"），只按高度定字号
    ln.size_px = 200 * scale
    # 原图的字体常常比 PPT 里的字体窄（标题用的压缩黑体）。同样字号下，记下每个字要收紧多少才能和原图一样宽
    n_gap = max(1, len(ln.text))
    ln.fit = float(np.clip((ink_w - (bx1 - bx0) * scale) / n_gap / max(1.0, ln.size_px), -0.05, 0.03))
    ln.baseline = ln.ink[1] - by0 * scale                     # by0 是负数（基线以上）

    # 粗细：量笔画宽度，再和参考字体的常规体 / 粗体在同一字号下的笔画宽度比较
    solid = dist > 0.5 * np.percentile(dist[ink], 90)
    sw = _stroke_width(solid & ink)
    ref = {}
    for bold in (False, True):
        fb = ref_font(200, bold)
        bb = fb.getbbox(ln.text, anchor="ls")
        tmp = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        ImageDraw.Draw(tmp).text((4 - bb[0], 4 - bb[1]), ln.text, font=fb, fill=255, anchor="ls")
        ref[bold] = _stroke_width(np.asarray(tmp) > 127) * scale
    ln.bold = sw > ref[False] ** 0.4 * ref[True] ** 0.6       # 阈值略偏向粗体一侧，减少误判
    if ln.size_px < BOLD_MIN_PX and not ln.badge:             # 字太小时笔画只有 1 像素左右，量不准，一律按常规体
        ln.bold = False                                       # （序号的字笔画粗、背景干净，可以量）

    # 一行里有两种颜色（比如重点词标成别的颜色）时，拆成几段分别上色
    ln.runs = _split_runs(ln, roi, ink, solid & ink, dist, x0, scale, bg)

    # 记录到整页掩膜里，向外扩一圈：压缩图片的文字周围有一圈"脏边"，不盖住会留下残影
    k = max(3, int(round(ln.size_px * 0.08)) | 1)             # 这里只贴着笔画留一点边，更宽的范围由 erase_text 处理
    m = cv2.dilate(ink.astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    full_mask[y0:y1, x0:x1] = np.maximum(full_mask[y0:y1, x0:x1], m)
    return True


def _split_runs(ln, roi, ink, solid, dist, x0, scale, bg):
    """返回 [(文字, 颜色, 是否加粗, 字号像素), ...]。只有一种颜色时就是一段。"""
    whole = [(ln.text, ln.color, ln.bold, ln.size_px)]
    strong = ink & (dist >= np.percentile(dist[ink], 40))
    px = roi[strong]
    if len(ln.text) < 2 or len(px) < 40:
        return whole
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px.astype(np.float32), 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=2) / float(len(px))
    if np.linalg.norm(cen[0] - cen[1]) < 45 or share.min() < 0.10:
        return whole                                           # 两类颜色太接近，或者其中一类太少：当作单色
    # 笔画边缘的抗锯齿像素是"文字色和背景色的混合"，会聚成一个假的浅色类。
    # 真正的第二种颜色不在"背景色 → 文字色"这条连线上，据此区分。
    v = [cen[0] - bg, cen[1] - bg]
    far = int(np.linalg.norm(v[1]) > np.linalg.norm(v[0]))
    axis = v[far] / (np.linalg.norm(v[far]) + 1e-6)
    near = v[1 - far]
    if np.linalg.norm(near - np.dot(near, axis) * axis) < 30:     # 黑字里的深绿重点词，偏离只有四五十，门槛不能太高
        return whole

    labmap = np.full(ink.shape, -1, np.int8)
    labmap[strong] = lab.ravel()
    f = ref_font(200)
    total = f.getlength(ln.text)
    if total <= 0:
        return whole
    ix0, ix1 = ln.ink[0] - x0, ln.ink[2] - x0
    n = len(ln.text)

    # 逐列看哪种颜色占多数，平滑后找颜色切换的位置
    col = np.full(ix1 - ix0, -1, int)
    for j in range(ix0, ix1):
        c0, c1 = int((labmap[:, j] == 0).sum()), int((labmap[:, j] == 1).sum())
        if c0 + c1:
            col[j - ix0] = 0 if c0 >= c1 else 1
    last = int(share.argmax())
    for j in range(len(col)):                                  # 空列沿用左边的颜色
        if col[j] < 0:
            col[j] = last
        last = col[j]
    k = max(3, int(ln.size_px * 0.6)) | 1
    sm = (cv2.blur(col.astype(np.float32).reshape(1, -1), (k, 1)).ravel() > 0.5).astype(int)
    cuts = [j for j in range(1, len(sm)) if sm[j] != sm[j - 1]]

    def ink_h(a, b):
        rows = np.where(ink[:, a:b].any(axis=1))[0]
        return (rows.max() - rows.min() + 1) if len(rows) else 0

    def ref_h(t):
        bb = f.getbbox(t, anchor="ls")
        return max(1, bb[3] - bb[1])

    if ln.chars and len(ln.chars) == n:
        # OCR 给了每个字的位置：直接逐字看颜色，最准
        char_lab, prev = [], int(share.argmax())
        for a_, b_ in ln.chars:
            seg = labmap[:, max(0, a_ - x0):max(a_ - x0 + 1, b_ - x0)]
            n0, n1_ = int((seg == 0).sum()), int((seg == 1).sum())
            if n0 + n1_ >= 4:
                prev = 0 if n0 >= n1_ else 1
            char_lab.append(prev)
        for i in range(1, n - 1):
            if char_lab[i - 1] == char_lab[i + 1] != char_lab[i]:
                char_lab[i] = char_lab[i - 1]
        if len(set(char_lab)) == 1:
            return whole
        pieces, start = [], 0
        for i in range(1, n + 1):
            if i == n or char_lab[i] != char_lab[start]:
                pieces.append((start, i, max(0, ln.chars[start][0] - x0), ln.chars[i - 1][1] - x0, char_lab[start]))
                start = i
        per_piece_size = True
    elif len(cuts) == 1:
        per_piece_size = True
        # 只有一处切换：两段的字号可能不同（重点词常常更大），用"各段高度 → 各段字宽"反推分界在第几个字
        xt = ix0 + cuts[0]
        L1, L2 = xt - ix0, ix1 - xt
        h1, h2 = ink_h(ix0, xt), ink_h(xt, ix1)
        best, n1 = 1e9, 1
        for i in range(1, n):
            t1, t2 = ln.text[:i], ln.text[i:]
            p1 = f.getlength(t1) * h1 / ref_h(t1)
            p2 = f.getlength(t2) * h2 / ref_h(t2)
            if p1 <= 0 or p2 <= 0:
                continue
            err = abs(np.log(p1 / L1)) + abs(np.log(p2 / L2))
            if err < best:
                best, n1 = err, i
        pieces = [(0, n1, ix0, xt, int(sm[0])), (n1, n, xt, ix1, int(sm[-1]))]
    else:
        per_piece_size = False
        # 多处切换：按参考字宽估每个字的位置，逐字投票
        char_lab, prev = [], int(share.argmax())
        for i in range(n):
            a = ix0 + int(round(f.getlength(ln.text[:i]) / total * (ix1 - ix0)))
            b = ix0 + int(round(f.getlength(ln.text[:i + 1]) / total * (ix1 - ix0)))
            seg = labmap[:, a:max(b, a + 1)]
            n0, n1_ = int((seg == 0).sum()), int((seg == 1).sum())
            if n0 + n1_ >= 4:
                prev = 0 if n0 >= n1_ else 1
            char_lab.append(prev)
        for i in range(1, n - 1):                              # 孤立的单个字并回去
            if char_lab[i - 1] == char_lab[i + 1] != char_lab[i]:
                char_lab[i] = char_lab[i - 1]
        if len(set(char_lab)) == 1:
            return whole
        pieces, start = [], 0
        for i in range(1, n + 1):
            if i == n or char_lab[i] != char_lab[start]:
                a = ix0 + int(round(f.getlength(ln.text[:start]) / total * (ix1 - ix0)))
                b = ix0 + int(round(f.getlength(ln.text[:i]) / total * (ix1 - ix0)))
                pieces.append((start, i, a, b, char_lab[start]))
                start = i

    runs = []
    for c0, c1, a, b, k_ in pieces:
        seg = ln.text[c0:c1]
        b_, g_, r_ = cen[k_]
        size = 200.0 * ink_h(a, b) / ref_h(seg) if per_piece_size and ink_h(a, b) else ln.size_px
        if not 0.6 * ln.size_px <= size <= 1.6 * ln.size_px:
            size = ln.size_px
        bold = ln.bold
        if size >= BOLD_MIN_PX and b - a > 4:                  # 每段单独判断粗细（"重点词加粗"很常见）
            sw = _stroke_width(solid[:, a:b])
            rw = {}
            for bd in (False, True):
                fb = ref_font(200, bd)
                bb = fb.getbbox(seg, anchor="ls")
                tmp = Image.new("L", (max(1, bb[2] - bb[0]) + 8, max(1, bb[3] - bb[1]) + 8), 0)
                ImageDraw.Draw(tmp).text((4 - bb[0], 4 - bb[1]), seg, font=fb, fill=255, anchor="ls")
                rw[bd] = _stroke_width(np.asarray(tmp) > 127) * size / 200.0
            if rw[False] > 0 and rw[True] > 0:
                bold = sw > rw[False] ** 0.4 * rw[True] ** 0.6
        runs.append((seg, (int(r_), int(g_), int(b_)), bool(bold), float(size)))
    return runs


def _slant(ink):
    """量一行字往右倾斜了多少。办法：把每一行像素按不同的斜率往回"扶正"，
    扶正到位时竖笔画会重新对齐成一列，按列统计的笔画量最集中。返回 (最佳斜率, 比不扶正时集中多少倍)。"""
    h, w = ink.shape
    if h < 10 or w < 4:
        return 0.0, 1.0, 0.0
    ys, xs = np.where(ink)
    up = (h - 1 - ys).astype(np.float32)
    def sharp(k):
        col = np.round(xs - k * up).astype(np.int64)
        return float((np.bincount(col - col.min()).astype(np.float64) ** 2).sum())
    base = sharp(0.0)
    best_k, best = 0.0, base
    for k in np.arange(-0.10, 0.42, 0.02):
        v = sharp(float(k))
        if v > best:
            best_k, best = float(k), v
    mom = float(np.cov(xs, up)[0, 1] / max(1e-6, up.var()))    # 另一种量法：笔画整体"越往上越偏右"的程度
    return best_k, best / max(base, 1.0), mom


def refit_widths(lines):
    """字号、粗细、补上的标点都定下来之后，重新算每行的字距：
    用参考字体把最终的文字排一遍，和原图里这行字的实际宽度比，差多少就靠字距收紧 / 放宽多少。
    目标宽度取原图的 98.5%：宁可略窄一点，也不要顶到旁边的分隔线、图片。"""
    for ln in lines:
        if ln.badge or not ln.ink or ln.size_px <= 0 or not ln.text:
            continue
        target = (ln.ink[2] - ln.ink[0]) * 0.985
        lead = len(ln.text) - len(ln.text.lstrip("—"))
        runs = ln.runs or [(ln.text, ln.color, ln.bold, ln.size_px)]
        width = 0.0
        for seg, _, bold, px in runs:
            if seg:
                width += ref_font(200, bold).getlength(seg) * px / 200.0
        n = max(1, len(ln.text))
        need = (target - width) / n / ln.size_px
        ln.fit = float(np.clip(need, -0.06, 0.01 if lead else 0.03))
        if need < -0.06:
            # 原图用的是更窄的字体。字距收到 6% 已经是极限（再收字就叠在一起了），剩下的差距靠把字号调小一点补上
            k = float(np.clip(target / (width + ln.fit * n * ln.size_px), 0.86, 1.0))
            ln.size_px *= k
            if ln.runs:
                ln.runs = [(t, c, b_, px * k) for t, c, b_, px in ln.runs]


def detect_italic(lines):
    """斜体判断。普通文字逐行判断；序号统一判断（同一页的序号用的是同一种字体，
    单个数字像"7""4"本身就有斜笔画，一个一个判断会出错，放在一起看多数）。"""
    def is_it(sl):
        return 0.09 <= sl[0] <= 0.36 and sl[1] >= 1.03
    for ln in lines:
        # 只对纯数字 / 英文的短文字判断（页码"06"这类）：汉字笔画横竖撇捺都有，量不准，而且中文很少用斜体
        if not ln.badge and ln.text.isascii() and ln.text.isalnum() and len(ln.text) <= 6:
            ln.italic = is_it(ln.slant)
    badges = [ln for ln in lines if ln.badge]
    if badges:
        yes = sum(is_it(ln.slant) for ln in badges) >= 0.6 * len(badges)
        for ln in badges:
            ln.italic = bool(yes)


def _stroke_width(binary):
    """笔画宽度 ≈ 2 × 面积 / 周长（细长笔画的近似）。先放大 3 倍，小字也量得准。"""
    big = cv2.resize(binary.astype(np.uint8) * 255, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    big = (big > 127).astype(np.uint8)
    contours, _ = cv2.findContours(big, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    if perim == 0:
        return 0.0
    return 2.0 * float(big.sum()) / perim / 3.0


# ----------------------------------------------------------------------------
# 第 3 步：抹字补背景
# ----------------------------------------------------------------------------
LAMA_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "lama_fp32.onnx"),
    os.path.expanduser("~/.cache/img2pptx/lama_fp32.onnx"),
]
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"   # 国内可换成 hf-mirror.com
_lama = {}


def erase_text(img_bgr, mask, method="auto", smooth_mask=None, hole_mask=None):
    """把 mask 标出的区域抹掉并补全背景。mask 是贴着笔画 / 图标的"紧"掩膜。

    lama  ：用 LaMa 修复模型，能把被文字压住的线条、纹理"续"上（需要模型文件）
    telea ：OpenCV 自带的快速算法，纯色 / 渐变背景够用，复杂背景会糊
    auto  ：找得到模型就用 lama，否则用 telea

    为了尽量少动原图，分两种范围：
      宽掩膜 —— 喂给补全算法，保证它参考到的边界像素是干净的（不带文字的脏边）
      紧掩膜 —— 真正替换回原图的范围，只比笔画宽 2 像素左右，边缘羽化；其余像素保持原图不变
    """
    H, W = mask.shape
    r = max(3, W // 250)
    wide = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    # 第一步：背景是纯色或平滑渐变的地方（标题色带、卡片底色……），直接"算"出底下的颜色，不交给模型画。
    # 模型画出来的东西总带一点模糊和色差，在纯色底上特别显眼；算出来的是干净的渐变，看不出痕迹。
    exact, solved = _fill_smooth(img_bgr, wide, hole_mask)
    rest = np.where(solved, 0, wide).astype(np.uint8)          # 剩下的才是背景复杂、需要模型补的

    how = "telea"
    pred = None
    if rest.any() and method in ("auto", "lama"):
        path = next((p for p in LAMA_PATHS if os.path.exists(p)), None)
        if path:
            base = exact                                       # 已经算好的地方先填上，模型参考到的就是干净的画面
            pred = _fix_artifacts(base, _erase_lama(base, rest, path), rest)
            how = "lama"
        elif method == "lama":
            raise RuntimeError("找不到 LaMa 模型，请下载 " + LAMA_URL + " 放到脚本旁的 models/ 文件夹")
    if pred is None:
        pred = cv2.inpaint(exact, rest, 3, cv2.INPAINT_TELEA) if rest.any() else exact
    if not rest.any():
        how = "直接计算"
    elif solved.any():
        how += " + 直接计算"

    paste = cv2.dilate(mask, np.ones((5, 5), np.uint8))        # 比笔画宽 2 像素
    paste[solved] = 255                                        # 算出来的地方整块替换，连最淡的残影也不留
    alpha = cv2.GaussianBlur(paste, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    alpha = np.maximum(alpha, (mask > 0).astype(np.float32)[..., None])   # 笔画本身必须完全替换
    alpha = np.maximum(alpha, solved.astype(np.float32)[..., None])
    out = pred.astype(np.float32) * alpha + img_bgr.astype(np.float32) * (1 - alpha)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    # 四周是干净底色的大块空洞（抠走插画后留下的）直接平滑填充：
    # 模型在这种大洞里喜欢"照着旁边的东西再画几个"（比如重复画箭头），反而添乱
    if smooth_mask is not None and (smooth_mask > 0)[~solved].any():
        sm = np.where(solved, 0, smooth_mask).astype(np.uint8)
        out = cv2.inpaint(out, cv2.dilate(sm, np.ones((7, 7), np.uint8)), 5, cv2.INPAINT_TELEA)
    return out, how


def _fill_from_edges(img_bgr, hole):
    """填一大块空洞（抠走照片、插画后留下的）。每个像素的颜色 = 它正上、正下、正左、正右
    四个方向上最近的洞外像素的加权平均，离得越近权重越大。结果是从四边向中间平滑过渡的颜色，
    不会像通用补全算法那样在大洞里留下沙漏形、放射状的痕迹，也不会凭空画出东西。"""
    out = img_bgr.copy()
    H, W = hole.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        p = 6
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        hsub = hole[y0:y1, x0:x1]
        comp = lab[y0:y1, x0:x1] == i
        src = cv2.blur(np.where(hsub[..., None], 0, out[y0:y1, x0:x1]).astype(np.float32), (5, 5))
        cnt = cv2.blur((~hsub).astype(np.float32), (5, 5))
        src = src / np.maximum(cnt, 1e-3)[..., None]              # 洞外像素的局部平均（洞里的不参与），避免拉出条纹
        hh, ww = hsub.shape
        cols = np.tile(np.arange(ww), (hh, 1))
        rows = np.tile(np.arange(hh)[:, None], (1, ww))
        valid = ~hsub & (cnt > 0.3)
        L = np.maximum.accumulate(np.where(valid, cols, -1), axis=1)
        R = np.minimum.accumulate(np.where(valid, cols, ww)[:, ::-1], axis=1)[:, ::-1]
        T = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
        B = np.minimum.accumulate(np.where(valid, rows, hh)[::-1], axis=0)[::-1]
        acc = np.zeros((hh, ww, 3), np.float32)
        wsum = np.zeros((hh, ww), np.float32)
        for idx, is_col, ok in ((L, True, L >= 0), (R, True, R < ww), (T, False, T >= 0), (B, False, B < hh)):
            d = np.abs((cols if is_col else rows) - idx).astype(np.float32)
            wgt = np.where(ok, 1.0 / np.maximum(d, 1.0) ** 2, 0.0)
            ci = np.clip(idx, 0, (ww if is_col else hh) - 1)
            val = src[rows, ci] if is_col else src[ci, cols]
            acc += val * wgt[..., None]
            wsum += wgt
        fill = acc / np.maximum(wsum, 1e-6)[..., None]
        k = max(5, (min(w, h) // 6) | 1)
        soft = cv2.GaussianBlur(np.where(comp[..., None], fill, out[y0:y1, x0:x1].astype(np.float32)), (k, k), 0)
        res = out[y0:y1, x0:x1].astype(np.float32)
        res[comp] = soft[comp]
        out[y0:y1, x0:x1] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
    return out


def _fill_smooth(img_bgr, wide, big_hint=None, max_rms=4.5):
    """对每一块要抹的区域：取它四周一圈的像素，拟合一个平滑的颜色曲面（二次多项式，能表示纯色和渐变）。
    如果四周的颜色和这个曲面几乎完全吻合，说明底下就是平滑背景，直接用曲面的值填进去。
    返回 (填好的图, 哪些像素是这样解决的)。"""
    H, W = wide.shape
    out = img_bgr.copy()
    solved = np.zeros((H, W), bool)
    # 抠走照片 / 插画留下的洞单独成块：它们常常和旁边的文字、图标的抹除范围连在一起，
    # 混在一起算的话，会把色带上图标的位置也填成卡片底色
    holes = np.zeros((H, W), bool)
    if big_hint is not None:
        holes = (cv2.dilate(big_hint, np.ones((9, 9), np.uint8)) > 0) & (wide > 0)
    n1, lab1, stats1, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(((wide > 0) & ~holes).astype(np.uint8), connectivity=8)
    lab = np.where(lab1 > 0, lab1, np.where(lab2 > 0, lab2 + n1 - 1, 0))
    stats = np.vstack([stats1, stats2[1:]])
    n = n1 + n2 - 1
    f = img_bgr.astype(np.float32)
    np.seterr(over='ignore', invalid='ignore', divide='ignore')   # macOS 上 numpy 的矩阵乘法会误报溢出警告，结果另有检查
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        p = 8
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        comp = lab[y0:y1, x0:x1] == i
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((2 * p - 3, 2 * p - 3), np.uint8)) > 0) & (wide[y0:y1, x0:x1] == 0)
        if ring.sum() < max(30, 0.5 * (2 * (w + h))):
            continue                                           # 四周能参考的像素太少（紧挨着别的要抹的东西）
        ys, xs = np.where(ring)
        sx, sy = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
        u, v = (xs - (x1 - x0) / 2.0) / sx, (ys - (y1 - y0) / 2.0) / sy
        A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1).astype(np.float64)
        px = f[y0:y1, x0:x1][ring].astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, px, rcond=None)
        if not np.isfinite(coef).all():
            continue
        res = px - A @ coef
        rms = float(np.sqrt((res ** 2).mean()))
        if rms > max_rms or float(np.percentile(np.abs(res).max(axis=1), 97)) > 4 * max_rms:
            # 四周不平滑。如果这是抠走照片 / 插画留下的大洞，再试一次：四周往往大部分是卡片底色，
            # 只有一边挨着别的东西（上面的标题色带、下面的标签）。只用占多数的那种颜色来算，
            # 填出来就是干净的卡片底色 —— 模型在这种大洞里容易画出沙漏形的痕迹或者重复的图案。
            is_big = i < n1                                    # 只有抠走图片留下的洞才这样处理
            if not is_big:
                continue                                       # 小块（文字、图标）：留给模型
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, kl, kc = cv2.kmeans(px.astype(np.float32), 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            cnt = np.bincount(kl.ravel(), minlength=3)
            # 候选：占比够大的类里，取最浅的（卡片底色一般比色带、照片边缘浅）
            cands = [k_ for k_ in range(3) if cnt[k_] >= 0.3 * len(px)]
            if not cands:
                continue
            dom = max(cands, key=lambda k_: float(kc[k_].sum()))
            inl = np.linalg.norm(px - kc[dom], axis=1) < 26
            if inl.sum() < 40:
                continue
            A2 = A[inl][:, :3]                                 # 大洞只用一次项（平面渐变），二次项外推太远不可靠
            coef2, *_ = np.linalg.lstsq(A2, px[inl], rcond=None)
            if not np.isfinite(coef2).all():
                continue
            coef = np.vstack([coef2, np.zeros((3, 3))])
            rms = float(np.sqrt(((px[inl] - A2 @ coef2) ** 2).mean()))
            px = px[inl]
            if rms > 2.5 * max_rms:
                continue
        cy, cx = np.where(comp)
        cu, cv_ = (cx - (x1 - x0) / 2.0) / sx, (cy - (y1 - y0) / 2.0) / sy
        B = np.stack([np.ones_like(cu), cu, cv_, cu * cu, cu * cv_, cv_ * cv_], axis=1).astype(np.float64)
        val = B @ coef
        if not np.isfinite(val).all():
            continue
        val = np.clip(val, 0, 255)
        # 外推保护：算出来的颜色不能超出四周颜色的范围太多
        lo, hi = px.min(axis=0) - 12, px.max(axis=0) + 12
        if (val < lo).any() or (val > hi).any():
            val = np.clip(val, lo, hi)
        # 原图的色带有细微的颗粒；填进去的如果绝对平滑，在大色块上反而像一块"补丁"。加上和四周同等强度的颗粒
        noise = np.random.default_rng(i).normal(0, min(rms, 2.5), val.shape)
        out[y0:y1, x0:x1][comp] = np.clip(val + noise + 0.5, 0, 255).astype(np.uint8)
        solved[y0:y1, x0:x1][comp] = True
    return out, solved


def _fix_artifacts(img_bgr, pred, wide, tol=42):
    """模型偶尔会凭空画出一小团周围根本没有的颜色。逐块检查：补出来的颜色如果在这块区域
    周边找不到相近的，就判为瑕疵，改用普通算法重填。"""
    n, lab = cv2.connectedComponents((wide > 0).astype(np.uint8), connectivity=8)
    bad = np.zeros(wide.shape, np.uint8)
    rng = np.random.default_rng(0)
    for i in range(1, n):
        comp = lab == i
        ys, xs = np.where(comp)
        y0, y1, x0, x1 = max(0, ys.min() - 6), ys.max() + 7, max(0, xs.min() - 6), xs.max() + 7
        c = comp[y0:y1, x0:x1]
        ring = (cv2.dilate(c.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & (wide[y0:y1, x0:x1] == 0)
        ctx = img_bgr[y0:y1, x0:x1][ring].astype(np.float32)
        if len(ctx) < 10:
            continue
        if len(ctx) > 600:
            ctx = ctx[rng.choice(len(ctx), 600, replace=False)]
        px = pred[y0:y1, x0:x1][c].astype(np.float32)
        dmin = np.full(len(px), 1e9, np.float32)
        for k in range(0, len(ctx), 100):                      # 分批算"到周边最近颜色的距离"
            d = np.linalg.norm(px[:, None, :] - ctx[None, k:k + 100, :], axis=2).min(axis=1)
            dmin = np.minimum(dmin, d)
        flag = np.zeros(c.shape, np.uint8)
        flag[c] = (dmin > tol).astype(np.uint8)
        flag = cv2.morphologyEx(flag, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # 去掉孤立的单个像素
        bad[y0:y1, x0:x1] |= flag
    if bad.any():
        bad = cv2.dilate(bad, np.ones((7, 7), np.uint8)) * 255
        bad &= wide                                           # 只在本来就要重画的范围内改
        pred = cv2.inpaint(pred, bad, 4, cv2.INPAINT_TELEA)
    return pred


def _erase_lama(img_bgr, mask, model_path, tile=512, margin=64):
    """模型固定吃 512×512，所以按原分辨率切块处理（不缩放，细节不丢），每块只取中间部分的结果。"""
    import onnxruntime as ort
    if "sess" not in _lama:
        _lama["sess"] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    sess = _lama["sess"]
    H0, W0 = mask.shape
    ph, pw = max(0, tile - H0), max(0, tile - W0)              # 图比 512 小时先镜像补边
    img = cv2.copyMakeBorder(img_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    todo = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0) > 0
    H, W = todo.shape
    out = img.copy()
    stride = tile - 2 * margin

    def starts(n):
        v = list(range(0, n - tile + 1, stride))
        if v[-1] != n - tile:
            v.append(n - tile)
        return v

    for ty in starts(H):
        for tx in starts(W):
            # 本块负责的"核心区"：去掉四周 margin（贴着图片边缘的那一侧不去）
            cy0 = ty + (margin if ty > 0 else 0)
            cx0 = tx + (margin if tx > 0 else 0)
            cy1 = ty + tile - (margin if ty + tile < H else 0)
            cx1 = tx + tile - (margin if tx + tile < W else 0)
            if not todo[cy0:cy1, cx0:cx1].any():
                continue
            m = todo[ty:ty + tile, tx:tx + tile]
            rgb = out[ty:ty + tile, tx:tx + tile, ::-1].astype(np.float32) / 255.0
            pred = sess.run(None, {
                "image": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]),
                "mask": m.astype(np.float32)[None, None]})[0][0].transpose(1, 2, 0)
            if pred.max() <= 1.5:
                pred = pred * 255.0
            pred = np.clip(pred, 0, 255).astype(np.uint8)[..., ::-1]
            core = np.zeros_like(m)
            core[cy0 - ty:cy1 - ty, cx0 - tx:cx1 - tx] = True
            sel = m & core
            out[ty:ty + tile, tx:tx + tile][sel] = pred[sel]
            todo[ty:ty + tile, tx:tx + tile][sel] = False
    return out[:H0, :W0]


# ----------------------------------------------------------------------------
# 照片 / 写实插画区域：颜色层次特别丰富的地方。里面的字属于画面本身（招牌、车身字），不转文本框
# ----------------------------------------------------------------------------
def find_picture_regions(img_bgr, text_mask):
    """返回 [(x, y, w, h), ...]。先把文字像素排除，再按小方块统计"有多少种颜色"。"""
    H, W = img_bgr.shape[:2]
    b = max(8, W // 100)
    q = (img_bgr >> 4).astype(np.int32)
    key = q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]
    free = cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) == 0
    gh, gw = H // b, W // b
    rich = np.zeros((gh, gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            fr = free[i * b:(i + 1) * b, j * b:(j + 1) * b].ravel()
            if fr.mean() < 0.4:
                continue
            blk = key[i * b:(i + 1) * b, j * b:(j + 1) * b].ravel()[fr]
            cnt = np.bincount(blk, minlength=4096)
            rich[i, j] = float((cnt > 0.015 * blk.size).sum() >= 13)
    dense = (cv2.blur(rich, (3, 3)) >= 0.45).astype(np.uint8)
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if min(w, h) * b >= 0.03 * W:
            out.append((int(x * b), int(y * b), int(w * b), int(h * b)))
    return out


# ----------------------------------------------------------------------------
# 图标提取：把颜色鲜明、大小适中的独立图形抠成透明背景的小图
# ----------------------------------------------------------------------------
@dataclass
class Icon:
    x: int
    y: int
    rgba: np.ndarray            # 透明背景的小图（BGRA）
    kind: str = "图标"          # "图标" 或 "图片"
    box: tuple = None           # 实际内容的包围盒 (x, y, w, h)，不含透明留边
    score: int = 100            # 置信分 0–100
    why: str = ""               # 分项得分说明，给页面显示用


def extract_icons(img_bgr, text_mask, exclude=None, rejected=None, avoid=None, pale_exclude=None):
    """返回 (图标列表, 需要从底图抹掉的掩膜)。

    思路：
      1. 前景 = 饱和度高或很深的像素（浅色背景上的图标基本都是这样），去掉文字
      2. 按色系分别找连通块，这样红色定位针不会和蓝色箭头粘在一起
      3. 只留"大小适中、长宽比正常、不贴边"的块 —— 页眉色带、长箭头、分隔线都会被排除
      4. 实心图标如果和细线（箭头）粘连，用开运算把细线断开后再取
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(int) * 2, hsv[..., 1], hsv[..., 2]
    fg = (sat > 110) | (val < 80)                              # 颜色鲜明，或者很深（亮橙、亮黄这类很亮的颜色也算）
    # 注意：这里不能先把文字从前景里挖掉。色带上有字时，一挖就把色带切成好几块，每块都会被当成"图标"。
    # 改成先找连通块，再把"主要由文字组成"的块丢掉
    tm = cv2.dilate(text_mask, np.ones((3, 3), np.uint8)) > 0

    min_side = max(12, int(W * ICON_MIN_RATIO))
    max_side = int(W * ICON_MAX_RATIO)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(2, W // 400) + 1,) * 2)
    families = [(hue < 45) | (hue >= 330), (hue >= 45) & (hue < 200), (hue >= 200) & (hue < 330)]

    def ok(x, y, w, h):
        return (min_side <= max(w, h) <= max_side and min(w, h) >= min_side * 0.6
                and max(w, h) / max(1, min(w, h)) <= 4
                and x > 1 and y > 1 and x + w < W - 1 and y + h < H - 1)

    masks = []                                                 # 每个图标一张整页大小的布尔掩膜
    too_big_all = np.zeros((H, W), bool)                       # 所有"太大的色块"（色带、底板）
    for fam in families:
        m = (fg & fam).astype(np.uint8)
        taken = np.zeros((H, W), bool)
        # 先找整块就合格的
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        too_big = np.zeros((H, W), bool)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if ok(x, y, w, h) and area >= 0.08 * w * h:
                masks.append(lab == i)
                taken |= lab == i
            elif max(w, h) > max_side:
                too_big |= lab == i
        # 再从"太大"的块里找被细线连着的实心图标
        too_big_all |= too_big
        solid = cv2.morphologyEx((too_big).astype(np.uint8), cv2.MORPH_OPEN, open_k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if ok(x, y, w, h) and area >= 0.45 * w * h and min(w, h) >= min_side * 1.5:
                grown = cv2.dilate((lab == i).astype(np.uint8), open_k) > 0
                masks.append(grown & too_big)

    masks = [mk for mk in masks if tm[mk].mean() < 0.3]         # 文字块不要

    # 第三类：深色形状上的浅色图标（标题色带里的白色小图标）。上面按"颜色鲜明"找，找不到白色的
    kk = (W // 55) | 1
    light = (val > 215) & (sat < 70)
    # "在深色底上"：周围一片里有三成以上是深色 / 鲜艳的像素。（不能用中值——图标中心周围白色占多数，会被判成"不在深色底上"而缺一块）
    dark = ((val < 185) | (sat > 95)).astype(np.float32)
    on_dark = cv2.blur(dark, (kk, kk)) > 0.3
    lw = (light & on_dark & (cv2.dilate(text_mask, np.ones((3, 3), np.uint8)) == 0)).astype(np.uint8)
    lw = cv2.morphologyEx(lw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))       # 一个图标的几笔先连起来
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lw, connectivity=8)
    light_masks = []                                           # 单独放：不和旁边的深色图标合并
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if ok(x, y, w, h) and area >= 0.1 * w * h and max(w, h) <= max_side * 0.6:
            light_masks.append((lab == i) & light)

    # 第四类：浅色底上的淡彩图标（卡片右侧那种浅绿色的装饰图标）。颜色不够鲜明，前面三类都找不到。
    # 找法：和"周围的底色"比。底色用大范围中值估计；比底色明显深一点、又不属于前面任何一类的像素就是它
    kb = (W // 9) | 1                                          # 范围要比最大的图标大得多，否则实心图标内部会被当成底色
    small = cv2.resize(img_bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)   # 缩小一半再做，快四倍
    bg_est = cv2.resize(cv2.medianBlur(small, (kb // 2) | 1), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    diff = np.linalg.norm(img_bgr.astype(np.float32) - bg_est, axis=2)
    bg_hsv = cv2.cvtColor(np.clip(bg_est, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    flat_bg = (bg_hsv[..., 2] > 225) & (bg_hsv[..., 1] < 40)   # 只在浅色、干净的底上找（照片、色带上不找）
    # 其他图标边缘的抗锯齿像素也"比底色深一点"，会形成一圈假的淡彩轮廓；离鲜明颜色 3 像素以内的不算
    near_vivid = cv2.dilate((fg | (light & on_dark)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    pale = (diff > 26) & flat_bg & ~fg & ~light & ~tm & ~near_vivid
    pale_u8 = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pale_u8, connectivity=8)
    pale_masks = []
    pale_lines = np.zeros((H, W), bool)                        # 卡片边框、分隔线：给淡彩图标算"独立"分时不能把它们算成杂色
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) > max_side or max(w, h) / max(1, min(w, h)) > 6:
            pale_lines |= lab == i
        if max(w, h) > max_side or max(w, h) < 0.5 * min_side or area < 12:
            continue
        if max(w, h) / max(1, min(w, h)) > 6:                  # 细长的线（卡片边框、分隔线）不要
            continue
        pale_masks.append((lab == i) & pale)
    pale_masks = _merge_close(pale_masks, gap=max(5, W // 140), max_side=max_side)
    def pale_ok(mk):
        x, y, w, h = cv2.boundingRect(mk.astype(np.uint8))
        if not ok(x, y, w, h) or mk.sum() < 0.18 * w * h:        # 太稀疏的（弧线、虚线箭头）不是图标
            return False
        px = img_bgr[mk].astype(np.float32)                    # 淡彩图标是单一颜色的；颜色杂的是阴影、渐变、照片的一角
        return (np.linalg.norm(px - np.median(px, axis=0), axis=1) < 38).mean() >= 0.8
    def pale_grow(mk, vivid=False):
        # 底色估计在页面装饰（顶部的波浪色带）附近会偏深，图标靠那一侧的部分就"不够深"而被漏掉。
        # 用图标自己的颜色往外长：和它同色、连在一起的像素补回来；长到碰到搜索范围的边说明连上了别的东西，放弃
        x, y, w, h = cv2.boundingRect(mk.astype(np.uint8))
        e = int(0.3 * min(w, h)) + 4
        x0, y0, x1, y1 = max(0, x - e), max(0, y - e), min(W, x + w + e), min(H, y + h + e)
        sub = mk[y0:y1, x0:x1]
        win = img_bgr[y0:y1, x0:x1].astype(np.float32)
        gray = win.sum(axis=2)
        local_bg = np.median(win[gray >= np.percentile(gray[~sub], 60)], axis=0)   # 这一小块里的底色：取最亮的那部分像素
        like = (np.linalg.norm(win - local_bg, axis=2) > 14) & ~tm[y0:y1, x0:x1]
        if not vivid:
            like &= ~near_vivid[y0:y1, x0:x1]
        like &= cv2.dilate(pale_lines[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
        n2, lab2 = cv2.connectedComponents((like | sub).astype(np.uint8), connectivity=8)
        ids = np.unique(lab2[sub])
        grown = np.isin(lab2, ids[ids > 0])
        if grown[0].any() or grown[-1].any() or grown[:, 0].any() or grown[:, -1].any():
            return mk
        if grown.sum() > (1.6 if vivid else 3.0) * sub.sum():
            return mk
        out = mk.copy()
        out[y0:y1, x0:x1] |= grown
        return out
    pale_masks = [pale_grow(mk) for mk in pale_masks if pale_ok(mk)]

    # 位置很近的碎块合并成一个图标（比如由几笔组成的线条图标）
    gap = max(3, W // 330)                                     # 合并距离取小：按最小单元识别，挨得很近的笔画才并成一个
    masks = _merge_close(masks, gap=gap, max_side=max_side)

    # 图标旁边没达到尺寸门槛的小零件（比如船下面的波浪线）也并进来
    used = np.zeros((H, W), bool)
    for mk in masks:
        used |= mk
    n, lab, stats, _ = cv2.connectedComponentsWithStats((fg & ~used).astype(np.uint8), connectivity=8)
    boxes = [cv2.boundingRect(mk.astype(np.uint8)) for mk in masks]
    base = [(bw, bh, int(mk.sum())) for (bx, by, bw, bh), mk in zip(boxes, masks)]   # 吸收前的原始尺寸
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) > max_side or area < 4 or tm[lab == i].mean() > 0.3:
            continue
        for j, (bx, by, bw, bh) in enumerate(boxes):
            dx = max(0, max(x, bx) - min(x + w, bx + bw))
            dy = max(0, max(y, by) - min(y + h, by + bh))
            ux, uy = min(x, bx), min(y, by)
            uw, uh = max(x + w, bx + bw) - ux, max(y + h, by + bh) - uy
            ow, oh, oarea = base[j]
            if (dx <= gap and dy <= gap and max(uw, uh) <= max_side
                    and uw <= 1.35 * ow + 2 and uh <= 1.35 * oh + 2 and area <= 0.3 * oarea):
                masks[j] = masks[j] | (lab == i)
                boxes[j] = (ux, uy, uw, uh)
                break

    covered = np.zeros((H, W), np.uint8)
    for mk in masks:                                           # 彩色图标连同内部（填洞后）的范围
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(covered, cs, -1, 255, cv2.FILLED)
    light_masks = [mk for mk in light_masks if (covered[mk] > 0).mean() < 0.5]   # 圆形图标中间的白色图案不重复抠
    light_masks = _merge_close(light_masks, gap=max(6, W // 110), max_side=int(max_side * 0.6))   # 线条图标的几笔并成一个
    n_dark = len(masks)                                        # 这之后的都是"深底上的浅色图标"
    masks = masks + light_masks
    n_pale = len(masks)                                        # 再之后的是"浅底上的淡彩图标"
    pale_masks = [mk for mk in pale_masks if (covered[mk] > 0).mean() < 0.3]
    masks = masks + pale_masks
    icons, fills, erase = [], [], np.zeros((H, W), np.uint8)
    for mi, mk in enumerate(masks):
        filled = np.zeros((H, W), np.uint8)                    # 填掉内部的洞：圆形图标里的白色图案也算图标
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cs, -1, 255, cv2.FILLED)
        filled = cv2.dilate(filled, np.ones((3, 3), np.uint8))  # 外扩 1 像素，把抗锯齿边带上
        x, y, w, h = cv2.boundingRect(filled)
        if not ok(x, y, w, h):
            continue
        # 真图标是独立摆在背景上的；照片 / 插画里的一块颜色四周还是画面内容，据此排除
        k = max(3, W // 250)
        rx0, ry0, rx1, ry1 = max(0, x - k - 4), max(0, y - k - 4), min(W, x + w + k + 4), min(H, y + h + k + 4)
        sub = filled[ry0:ry1, rx0:rx1]
        # 只看紧贴着它的一圈（外扩 2–5 像素）：圈取宽了会把旁边的文字、色带边缘算进来，拉低"独立"分
        ring = (cv2.dilate(sub, np.ones((15, 15), np.uint8)) > 0) & (cv2.dilate(sub, np.ones((7, 7), np.uint8)) == 0)
        ring &= cv2.dilate(text_mask[ry0:ry1, rx0:rx1], np.ones((3, 3), np.uint8)) == 0
        if mi >= n_pale:
            ring &= cv2.dilate(pale_lines[ry0:ry1, rx0:rx1].astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
        rpx = img_bgr[ry0:ry1, rx0:rx1][ring].astype(np.float32)
        if exclude is not None and exclude[y + h // 2, x + w // 2]:
            continue                                           # 在已经抠出的照片 / 插画里面：它是画面的一部分
        score, why = score_icon(img_bgr, filled, text_mask, rpx)
        if mi >= n_pale:                                       # 淡彩图标：颜色淡，更容易把背景的渐变、阴影误认进来，条件卡严
            # "完整"这一项对它不适用（卡片边框也是同样的浅绿色，会被误判成"从大色块上切下来的"），只看另外三项
            p3 = [int(v) for v in __import__("re").findall(r"(?:独立|轮廓|内容) (\d+)", why)]
            if os.environ.get("PALEDBG"):
                cv2.imwrite("/tmp/pale_%d_%d.png" % (x, y), filled[max(0,y-6):y + h+6, max(0,x-6):x + w+6])
            if len(p3) == 3:
                # 淡彩图标和底色差得少，边缘的颜色变化本来就弱，"轮廓"按更低的门槛重算
                p3[1] = int(round(100 * _ramp(_contour_on_edges(img_bgr, filled, 8), 0.86, 0.985)))
                why = __import__("re").sub(r"轮廓 \d+", "轮廓 %d" % p3[1], why)
                score = int(round(0.65 * min(p3) + 0.35 * sum(p3) / 3))
            if pale_exclude is not None and pale_exclude[y + h // 2, x + w // 2]:
                score, why = min(score, 20), why + " · 在照片区域里"
            if score < 70:
                score, why = min(score, 40), why + " · 淡彩图标要求 70 分以上"
            else:
                why += " · 淡彩图标"
        elif mi >= n_dark:                                     # 浅色图标：四周得真的是深色底，否则只是浅色背景的一块
            rh = cv2.cvtColor(np.clip(rpx, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            darkfrac = float(((rh[:, 2] < 185) | (rh[:, 1] > 95)).mean()) if len(rh) else 0.0
            if darkfrac < 0.7:
                score, why = min(score, int(40 * darkfrac)), why + " · 四周不是深色底"
        # 紧贴着文字的色块：多半是文字底下那条色带被字隔出来的一段，不是图标（真图标和旁边的字之间总有一点空隙）
        near_ring = (cv2.dilate(sub, np.ones((13, 13), np.uint8)) > 0) & (sub == 0)
        touch = float(tm[ry0:ry1, rx0:rx1][near_ring].mean()) if near_ring.any() else 0.0
        if touch > 0.05 and max(w, h) > 1.6 * min_side:
            score, why = min(score, int(40 * (1 - _ramp(touch, 0.05, 0.15)) + 10)), why + " · 紧贴文字（是文字底下的色带）"
        # 它的边界外面如果还连着同一个大色块，说明它是从色带 / 底板上"断"下来的一段
        border = (cv2.dilate(sub, np.ones((7, 7), np.uint8)) > 0) & (sub == 0)
        joined = float(too_big_all[ry0:ry1, rx0:rx1][border].mean()) if border.any() else 0.0
        if joined > 0.1 and too_big_all[ry0:ry1, rx0:rx1][sub > 0].mean() > 0.5:   # 它自己也得是那个大色块的一部分（色带上的白图标不算）
            score, why = min(score, int(30 * (1 - _ramp(joined, 0.1, 0.25)) + 16)), why + " · 和一个大色块连着（是色带的一段）"
        if avoid is not None and (avoid[filled > 0] > 0).mean() > 0.25:
            score, why = min(score, 20), "和序号圆圈 / 圆形图标重叠：那边已经按整个圆处理了"
        if mi < n_dark and score >= ICON_KEEP_SCORE and flat_bg[filled > 0].mean() > 0.5:
            # 彩色图标上颜色很淡的部分（浅粉色的侧面、淡淡的描边）不够"鲜明"，会被落下，留在底图上成了残影；补回来
            g = pale_grow(filled > 0, vivid=True)
            if g.sum() > (filled > 0).sum():
                filled = cv2.morphologyEx(g.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                x, y, w, h = cv2.boundingRect(filled)
        p = 2
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        alpha = cv2.GaussianBlur(filled[y0:y1, x0:x1], (3, 3), 0)
        rgba = _flat_icon_rgba(img_bgr[y0:y1, x0:x1], filled[y0:y1, x0:x1], rpx)
        if rgba is None:
            rgba = np.dstack([img_bgr[y0:y1, x0:x1], alpha])
        ic = Icon(x0, y0, rgba, "图标", (x, y, w, h), score, why)
        if score < 35:
            if rejected is not None and score >= 15:
                rejected.append(ic)                            # 分数不够：不抠，但记下来给页面显示
            continue
        if mi >= n_pale:
            extra = np.zeros((H, W), np.uint8)
            extra[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = 255
            filled = np.maximum(filled, np.where((diff > 12) & flat_bg & ~tm, extra, 0).astype(np.uint8))
        icons.append(ic)                                       # 35 分以上先都留着：判断"谁包着谁"时，差一点及格的也要算进来
        fills.append(filled)
    # 一个候选如果把另一个合格的候选整个包在里面，它多半只是"容器"（色带的一段、底板），真正的图标是里面那个
    def inside_frac(a, b_):
        iw = max(0, min(a[0] + a[2], b_[0] + b_[2]) - max(a[0], b_[0]))
        ih = max(0, min(a[1] + a[3], b_[1] + b_[3]) - max(a[1], b_[1]))
        return iw * ih / float(a[2] * a[3])
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    busy_map = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1) ** 2) > 40
    dropped = {}
    for i, ic in enumerate(icons):
        area = ic.box[2] * ic.box[3]
        inner = [j for j, o in enumerate(icons)
                 if j != i and inside_frac(o.box, ic.box) > 0.4 and o.box[2] * o.box[3] <= 0.6 * area
                 and "淡彩图标要求" not in o.why]         # 没及格的淡彩候选不算数
        if not inner:
            continue
        # 把里面那些小候选的位置挖掉，看外面这个还剩不剩图案
        rest = cv2.erode(fills[i], np.ones((9, 9), np.uint8)) > 0
        for j in inner:
            x, y, w, h = icons[j].box
            rest[max(0, y - 7):y + h + 7, max(0, x - 7):x + w + 7] = False
        rest &= fg                                             # 只看它自己有颜色的像素（填洞填进来的白底不算）
        if rest.sum() < 20:                                    # 除了里面那些就不剩什么了：它就是这些零件的总和
            for j in inner:
                dropped[j] = "是旁边那个图标的一部分"
            continue
        # 颜色一样 → 里面的是它的零件（红色不等号里的横杠）；颜色不同 → 外面的是容器（绿色色带上的白色图标）
        own = np.median(img_bgr[rest].astype(np.float32), axis=0)
        same = [j for j in inner
                if np.linalg.norm(np.median(img_bgr[fills[j] > 0].astype(np.float32), axis=0) - own) < 70]
        if len(same) == len(inner):
            for j in inner:
                dropped[j] = "是旁边那个图标的一部分"
        elif not same and any((fills[j] > 0).sum() < 0.86 * icons[j].box[2] * icons[j].box[3] for j in inner):
            dropped[i] = "里面包着另一个图标，它只是容器"   # 里面确实有个"有形状"的图标
        elif not same:
            for j in inner:                                    # 里面只是些实心小方块：那是镂空图形（比如不等号）笔画之间的空隙
                dropped[j] = "是旁边那个图标的空隙"
    keep = []
    for i, ic in enumerate(icons):
        if i in dropped:
            ic.score, ic.why = min(ic.score, 40), dropped[i]
            if rejected is not None:
                rejected.append(ic)
        elif ic.score < ICON_KEEP_SCORE:
            if rejected is not None:
                rejected.append(ic)
        else:
            keep.append(i)
    for i in keep:
        erase = np.maximum(erase, cv2.dilate(fills[i], np.ones((7, 7), np.uint8)))
    icons = [icons[i] for i in keep]
    return icons, erase


# ----------------------------------------------------------------------------
# 图片提取：照片和写实插画。它们颜色杂，不能像图标那样按颜色找，
# 改用"内容块"的思路：先找出大片平坦的背景，剩下的实心块就是内容
# ----------------------------------------------------------------------------
def _flat_graphics(img_bgr, text_mask, edge):
    """找出纯色的图形件，返回掩膜（255 = 图形件）。两类：
      1. 又细又长的横条 / 竖条（时间轴、进度条）
      2. 里面压着文字的纯色色块（色带、胶囊标签、笔刷底纹）"""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    vivid = ((hsv[..., 1] > 95) & (hsv[..., 2] > 50)).astype(np.uint8)
    f = img_bgr.astype(np.float32)
    out = np.zeros((H, W), np.uint8)

    def uniform(sel, tol):
        px = f[sel]
        return len(px) >= 20 and (np.linalg.norm(px - np.median(px, axis=0), axis=1) < tol).mean() >= 0.7

    # 1. 细长条：先找"连续很长一段都是鲜明颜色"的行，再要求它是薄薄一条、颜色单一、里面没有纹理
    T = max(6, int(0.03 * H))                                  # 条的最大厚度
    vivid_all = vivid
    vivid = ((hsv[..., 1] > 55) & (hsv[..., 2] > 50)).astype(np.uint8)     # 找条时放宽一点：条的一头常常渐变变淡
    for horizontal in (True, False):
        L = max(40, (W if horizontal else H) // 12)
        run = cv2.morphologyEx(vivid, cv2.MORPH_OPEN, np.ones((1, L) if horizontal else (L, 1), np.uint8))
        # 比 T 厚的部分不是"条"（照片里的大片红色、条上串着的圆点）：减掉
        thick_part = cv2.morphologyEx(run, cv2.MORPH_OPEN, np.ones((T + 1, 1) if horizontal else (1, T + 1), np.uint8))
        thin = run & (1 - thick_part)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
        bars, boxes = np.zeros((H, W), np.uint8), []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            thick, length = (h, w) if horizontal else (w, h)
            if thick < max(8, 0.009 * H) or length < 8 * thick or area < 0.5 * w * h:
                continue                                       # 太细的是线条（车厢栏杆、表格线），不是时间轴
            # 条的上下两侧得是空的：照片里的一道红色（车厢栏板）上下还是画面
            sel = (lab == i).astype(np.uint8)
            grow = cv2.dilate(sel, np.ones((13, 1) if horizontal else (1, 13), np.uint8)) & (1 - cv2.dilate(sel, np.ones((5, 1) if horizontal else (1, 5), np.uint8)))
            if grow.sum() and vivid[grow > 0].mean() > 0.25:
                continue
            bars[sel > 0] = 1
            boxes.append((x, y, w, h))
            if horizontal:
                # 条的一头常常渐变得很淡：沿着同一行往两边接着找，只要还"有点颜色"就算条的一部分
                colsat = hsv[y:y + h, :, 1].astype(np.float32).mean(axis=0)
                on = colsat > 20
                x0, x1, gap = x, x + w, 0
                for xx in range(x - 1, max(-1, x - 2 * w - 1), -1):     # 允许跨过一小段空隙（圆点外面那圈白边）
                    gap = 0 if on[xx] else gap + 1
                    if gap > T:
                        break
                    if on[xx]:
                        x0 = xx
                gap = 0
                for xx in range(x + w, min(W, x + 3 * w)):
                    gap = 0 if on[xx] else gap + 1
                    if gap > T:
                        break
                    if on[xx]:
                        x1 = xx + 1
                seg = np.zeros((h, x1 - x0), np.uint8)
                seg[:, on[x0:x1]] = 1
                bars[y:y + h, x0:x1] |= seg
        if not boxes:
            continue
        out[bars > 0] = 255
        # 条上串着的节点（圆点）：个头小、颜色鲜明、中心落在条所在的那一行（列）上
        rest = vivid & (1 - bars)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        lo = min(b[0] if horizontal else b[1] for b in boxes) - 2 * T
        hi = max(b[0] + b[2] if horizontal else b[1] + b[3] for b in boxes) + 2 * T
        for q in range(1, n2):
            x, y, w, h, area = st2[q]
            if max(w, h) > 2.2 * T or area < 12:
                continue
            c_along, c_across = (x + w / 2, y + h / 2) if horizontal else (y + h / 2, x + w / 2)
            if not lo <= c_along <= hi:
                continue
            if not any(abs(c_across - ((b[1] + b[3] / 2) if horizontal else (b[0] + b[2] / 2))) <= 0.5 * (b[3] if horizontal else b[2]) + 2 for b in boxes):
                continue
            cs, _ = cv2.findContours((lab2[y:y + h, x:x + w] == q).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            patch = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(patch, cv2.convexHull(np.vstack(cs)), 255)      # 连同圆点中间的白心
            out[y:y + h, x:x + w] = np.maximum(out[y:y + h, x:x + w], patch)
    bars_only = cv2.dilate(out, np.ones((5, 5), np.uint8))
    vivid = vivid_all
    labels = []
    # 2. 压着文字的纯色色块：把文字盖住的地方也算进色块（否则色块被字切成碎片），再看它是不是单一颜色
    tm = cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats((vivid.astype(bool) | tm).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 0.6 * W or h > 0.2 * H or area < 200:
            continue
        sel = lab == i
        t_share = float(tm[sel].mean())
        body = sel & ~tm
        if not 0.12 <= t_share <= 0.75 or body.sum() < 80:
            continue                                           # 里面没有字，或者几乎全是字（那是彩色的文字本身）
        if not uniform(body, 60) or edge[cv2.erode(body.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0].mean() > 0.3:
            continue                                           # 颜色杂、有纹理：是照片上的字，不是色块
        out[sel] = 255
        labels.append((sel, np.median(f[body], axis=0)))       # 抹这上面的字时只参考色块自己的颜色（见 process_image）
    _flat_graphics.labels = labels
    _flat_graphics.bars = bars_only                            # 时间轴本身（给找图标的那一步用：轴上的圆点不是图标）
    return cv2.dilate(out, np.ones((7, 7), np.uint8))


def extract_pictures(img_bgr, text_mask, icons):
    """返回 (图片列表, 需要从底图抹掉的掩膜, 每张图片的整页布尔掩膜列表, 适合平滑填充的掩膜)。"""
    H, W = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.sqrt(gx ** 2 + gy ** 2).max(axis=2) / 4.0 > 10).astype(np.uint8)
    flat = (cv2.dilate(edge, np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=4)
    big = stats[:, cv2.CC_STAT_AREA] >= 0.008 * H * W          # 面积够大的平坦区域 = 背景 / 色块
    big[0] = False
    content = (~big[lab]).astype(np.uint8)
    content_nt = content.copy()                                # 不把文字算进去的版本：收紧矩形时用
    content[cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) > 0] = 1   # 文字算内容，免得在物体中间挖洞
    # 纯色的"图形件"不是配图的一部分：时间轴的横条、文字底下的色带 / 胶囊标签。
    # 它们常常紧挨着照片，不先拿掉的话会和照片粘成一块，连同旁边的圆点、标签一起被抠走
    graphic = _flat_graphics(img_bgr, text_mask, edge)
    content[graphic > 0] = 0
    content_nt[graphic > 0] = 0
    bars_core = _flat_graphics.bars

    edge_nt = edge.copy()                                      # 去掉文字之后的边缘图，用来量"细节密度"
    edge_nt[cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) > 0] = 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    agx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) / 4.0
    agy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    # "有细节的地方"：周围一小片里边缘够多。照片内部处处如此；卡片底色、留白（哪怕带渐变）不是
    kd = (W // 110) | 1
    detail = ((cv2.blur(edge_nt.astype(np.float32), (kd, kd)) > 0.08) & (content_nt > 0)).astype(np.uint8)
    k = max(4, W // 160)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    core = cv2.morphologyEx(content, cv2.MORPH_OPEN, ker)      # 开运算：去掉边框线、箭头这类细东西
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)

    def filled_of(mask_u8):
        out = np.zeros((H, W), np.uint8)
        cs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, 255, cv2.FILLED)
        return out

    def ring_of(filled, x, y, w, h, width):
        x0, y0, x1, y1 = max(0, x - width - 4), max(0, y - width - 4), min(W, x + w + width + 4), min(H, y + h + width + 4)
        sub = filled[y0:y1, x0:x1]
        ring = (cv2.dilate(sub, np.ones((2 * width + 5, 2 * width + 5), np.uint8)) > 0) & (cv2.dilate(sub, np.ones((5, 5), np.uint8)) == 0)
        return img_bgr[y0:y1, x0:x1][ring].astype(np.float32)

    pics, erase, masks = [], np.zeros((H, W), np.uint8), []
    smooth = np.zeros((H, W), np.uint8)                        # 四周很干净、适合直接平滑填充的空洞

    # ---- 方法一：内容块（照片、不规则的插画、设备图等） ----
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if min(w, h) < PIC_MIN_RATIO * W or max(w, h) > 0.5 * W:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        blob = ((cv2.dilate((lab == i).astype(np.uint8), ker) > 0) & (content > 0)).astype(np.uint8)

        # 照片常常放在一块浅色底板上，底板被照片切成窄条后也会被算进来。
        # 一层层剥：如果这块东西最外圈基本是同一种平坦颜色，就把这种颜色去掉再看
        for _ in range(3):
            band = cv2.erode(blob, np.ones((7, 7), np.uint8)) & (cv2.erode(blob, np.ones((2 * k + 7, 2 * k + 7), np.uint8)) == 0)
            inner = band & flat                                # 外圈那一带里的平坦像素（跳过最外 3 像素的描边）
            if band.sum() == 0 or inner.sum() < 0.4 * band.sum():
                break
            # 去掉"和这圈平坦像素连成一片"的平坦区域（底板可能带渐变，所以按连通而不是按颜色）
            m3, lab3 = cv2.connectedComponents((flat & blob).astype(np.uint8), connectivity=4)
            touch = np.unique(lab3[inner > 0])
            same = np.isin(lab3, touch[touch > 0])
            peeled = cv2.morphologyEx((blob & ~same).astype(np.uint8), cv2.MORPH_OPEN, ker)
            m2, lab2, st2, _ = cv2.connectedComponentsWithStats(peeled, connectivity=8)
            if m2 < 2:
                break
            j = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
            # 只往回长 3 像素（补回被开运算磨掉的边），长多了会把隔着一条缝的底板描边又粘回来
            nb = ((cv2.dilate((lab2 == j).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & (blob > 0) & ~same).astype(np.uint8)
            if nb.sum() < 0.3 * blob.sum():
                break
            blob = nb
        filled = filled_of(blob)
        x, y, w, h = cv2.boundingRect(filled)

        # 修边：从四边往里，把"几乎没有细节"的行 / 列去掉 —— 照片上方粘着的标题条、下方搭着的标签条都属于这种。
        # 细节密度按去掉文字后的边缘算；最多修掉一半，修多了说明它本来就不是一张图
        sub_f = filled[y:y + h, x:x + w] > 0
        sub_e = (edge_nt[y:y + h, x:x + w] > 0) & sub_f
        rd = cv2.blur((sub_e.sum(axis=1) / np.maximum(1, sub_f.sum(axis=1))).astype(np.float32).reshape(-1, 1), (1, 5)).ravel()
        cd = cv2.blur((sub_e.sum(axis=0) / np.maximum(1, sub_f.sum(axis=0))).astype(np.float32).reshape(-1, 1), (1, 5)).ravel()
        r_ok, c_ok = np.where(rd >= 0.12)[0], np.where(cd >= 0.12)[0]
        if len(r_ok) and len(c_ok):
            ty0, ty1, tx0, tx1 = r_ok.min(), r_ok.max() + 1, c_ok.min(), c_ok.max() + 1
            # 只修"成条"的稀疏边（至少占这一边长度的 8%）；零星几像素不动，免得把插画的边角削掉
            if ty0 < 0.08 * h: ty0 = 0
            if h - ty1 < 0.08 * h: ty1 = h
            if tx0 < 0.08 * w: tx0 = 0
            if w - tx1 < 0.08 * w: tx1 = w
            if ((ty1 - ty0) * (tx1 - tx0) >= 0.5 * w * h and ((ty1 - ty0) < h - 6 or (tx1 - tx0) < w - 6)
                    and min(ty1 - ty0, tx1 - tx0) >= PIC_MIN_RATIO * W):
                keep = np.zeros((H, W), np.uint8)
                keep[y + ty0:y + ty1, x + tx0:x + tx1] = filled[y + ty0:y + ty1, x + tx0:x + tx1]
                filled = keep
                x, y, w, h = cv2.boundingRect(filled)

        # 矩形照片：只留"满宽 / 满高"的行列，把搭在边上的标签条之类的东西切掉
        rows = (filled[y:y + h, x:x + w] > 0).sum(axis=1)
        cols = (filled[y:y + h, x:x + w] > 0).sum(axis=0)
        # 只对"本来就是矩形"的东西做：一半以上的行、列都接近最大宽度 / 高度。插画轮廓不规则，不能这样切
        rmax, cmax = np.percentile(rows, 90), np.percentile(cols, 90)   # 用 90 分位当"满宽"，不被个别毛刺行带偏
        rect_like = (rows >= 0.97 * rmax).mean() >= 0.5 and (cols >= 0.97 * cmax).mean() >= 0.5
        if rect_like:
            area_all = float((filled > 0).sum())
            best = None
            for axis in (0, 1):                                # 分别试"只切上下"和"只切左右"，取保留面积大的
                prof = rows if axis == 0 else cols
                keep = np.where(prof >= 0.9 * (rmax if axis == 0 else cmax))[0]
                if len(keep) < 4:
                    continue
                part = np.zeros((H, W), np.uint8)
                if axis == 0:
                    part[y + keep.min():y + keep.max() + 1, x:x + w] = filled[y + keep.min():y + keep.max() + 1, x:x + w]
                else:
                    part[y:y + h, x + keep.min():x + keep.max() + 1] = filled[y:y + h, x + keep.min():x + keep.max() + 1]
                bx, by, bw, bh = cv2.boundingRect(part)
                solid = (part[by:by + bh, bx:bx + bw] > 0).mean() if bw and bh else 0
                kept_area = float((part > 0).sum())
                # 真正的矩形照片，留下来那部分的两条侧边是笔直的；插画（比如两边有树的房子）不是
                pm = part[by:by + bh, bx:bx + bw] > 0
                pm = pm if axis == 0 else pm.T
                mid = pm[int(len(pm) * 0.1):max(int(len(pm) * 0.9), int(len(pm) * 0.1) + 1)]
                first = mid.argmax(axis=1)
                last = mid.shape[1] - 1 - mid[:, ::-1].argmax(axis=1)
                straight = first.std() <= 2.0 and last.std() <= 2.0
                # 被切掉的那块应该是单一颜色的东西（标签条、色带）；颜色杂说明它是图的一部分（比如货车车头）
                cut = (filled > 0) & (part == 0) & (text_mask == 0)
                if cut.sum() >= 20:
                    cpx = img_bgr[cut].astype(np.float32)
                    straight = straight and (np.linalg.norm(cpx - np.median(cpx, axis=0), axis=1) < 40).mean() >= 0.55
                if (straight and solid >= 0.93 and 0.7 * area_all <= kept_area < 0.97 * area_all
                        and min(bw, bh) >= PIC_MIN_RATIO * W and (best is None or kept_area > best[0])):
                    best = (kept_area, part)
            if best:
                filled = best[1]
                x, y, w, h = cv2.boundingRect(filled)

        # 去掉挂在图上的细东西：卡片的边框线、图下方说明文字的碎片。只留最大的那一整块
        fm = (filled > 0).astype(np.uint8)
        body = cv2.morphologyEx(fm, cv2.MORPH_OPEN, ker)
        m3, lab3, st3, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
        if m3 >= 2:
            j = 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))
            main = ((cv2.dilate((lab3 == j).astype(np.uint8), ker) > 0) & (fm > 0)).astype(np.uint8)
            m4, lab4 = cv2.connectedComponents(main, connectivity=8)      # 回长时可能又碰到细线，再取一次和主体相连的部分
            ids = np.unique(lab4[lab3 == j])
            main = np.isin(lab4, ids[ids > 0]).astype(np.uint8) * 255
            main = cv2.morphologyEx(main, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            if main.sum() >= 0.5 * 255 * fm.sum():
                filled = main
                x, y, w, h = cv2.boundingRect(filled)

        # 收边：配图常常放在一张白色卡片里，卡片的留白和边框线也会被算进来。
        # 如果这一块最外面一圈基本是同一种颜色，就把这种颜色去掉，只留里面真正的图
        for _ in range(2):
            fm = (filled > 0).astype(np.uint8)
            # 取"往里 2 像素到 6 像素"这一圈（最外 2 像素是抗锯齿的过渡色，不算）
            band = (cv2.erode(fm, np.ones((5, 5), np.uint8)) > 0) & (cv2.erode(fm, np.ones((13, 13), np.uint8)) == 0)
            if band.sum() < 40:
                break
            bc = np.median(f[band], axis=0)
            if os.environ.get("PICDBG"):
                print("     tighten?", (x, y, w, h), "band uniform", round(float((np.linalg.norm(f[band] - bc, axis=1) < 22).mean()), 2), bc.astype(int))
            if (np.linalg.norm(f[band] - bc, axis=1) < 22).mean() < 0.6:
                break                                          # 外圈颜色很杂：图本身已经顶到边了，不用收
            inner = ((fm > 0) & (np.linalg.norm(f - bc, axis=2) > 28)).astype(np.uint8)
            inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            m3, lab3, st3, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
            if m3 < 2:
                break
            j = 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))
            tight = filled_of((lab3 == j).astype(np.uint8))
            hull = np.zeros((H, W), np.uint8)
            cs, _ = cv2.findContours(tight, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(cs)), 255)
            hull &= filled
            if (hull > 0).sum() < 1.1 * (tight > 0).sum():
                tight = hull                                   # 本来就是凸的形状（圆角矩形照片）：用凸包，边更整齐
            if os.environ.get("PICDBG"):
                print("       ->", cv2.boundingRect(tight), "keep", round(float((tight > 0).sum()) / float(fm.sum()), 2))
            if not 0.4 * fm.sum() <= (tight > 0).sum() < 0.97 * fm.sum():
                break
            if min(cv2.boundingRect(tight)[2:]) < PIC_MIN_RATIO * W * 0.85:
                break
            filled = tight
            x, y, w, h = cv2.boundingRect(filled)

        # 如果它本来就是一张矩形照片（收紧后的矩形里几乎全是内容，且占了原来那块的大部分），就用规整的矩形
        (sx, sy, sw, sh), sides = _snap_photo_rect(agx, agy, (x, y, w, h))
        sfill = float(detail[sy:sy + sh, sx:sx + sw].mean()) if sides >= 3 else 0.0
        inside_share = (filled[sy:sy + sh, sx:sx + sw] > 0).sum() / float(max(1, (filled > 0).sum()))
        if os.environ.get("PICDBG"):
            print("     snap", (x, y, w, h), "->", (sx, sy, sw, sh), "fill", round(sfill, 2),
                  "sides", sides, "mask in rect", round(float((filled[sy:sy + sh, sx:sx + sw] > 0).mean()), 2), "share", round(inside_share, 2))
        # 条件：矩形里几乎全是内容；原来那块几乎全在矩形里（否则会把插画的屋顶、招牌切掉）
        # 真照片的四个角也是画面；抠好的实物图（白色货车）外形接近矩形，但四个角是底色，不能按矩形照片处理
        c = max(4, min(sw, sh) // 12)
        corners = [content_nt[sy:sy + c, sx:sx + c], content_nt[sy:sy + c, sx + sw - c:sx + sw],
                   content_nt[sy + sh - c:sy + sh, sx:sx + c], content_nt[sy + sh - c:sy + sh, sx + sw - c:sx + sw]]
        photo_corners = sum(float(q.mean()) > 0.5 for q in corners if q.size) >= 3
        if (sides >= 3 and sfill >= 0.6 and min(sw, sh) >= PIC_MIN_RATIO * W * 0.85 and photo_corners
                and (filled[sy:sy + sh, sx:sx + sw] > 0).mean() >= 0.88 and inside_share >= 0.9):
            filled = _rect_mask(img_bgr, sx, sy, sw, sh)
            x, y, w, h = sx, sy, sw, sh

        if (filled[bars_core > 0] > 0).any():                  # 图的阴影搭到时间轴上时：时间轴那几像素不归这张图
            filled = filled.copy()
            filled[bars_core > 0] = 0
            filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            x, y, w, h = cv2.boundingRect(filled)
        if os.environ.get("PICDBG"):
            print("   cand", (x, y, w, h), "edge", round(float(edge[filled > 0].mean()), 2), "fill", round(float((filled[y:y + h, x:x + w] > 0).mean()), 2))
        if min(w, h) < PIC_MIN_RATIO * W * 0.85:
            continue
        sel = filled > 0
        if (filled[y:y + h, x:x + w] > 0).mean() < 0.6:        # 形状太稀疏
            continue
        if edge[sel].mean() < 0.42:                            # 细节不够多：是色块或"底板 + 图标 + 标签"的卡片，不是图
            continue
        if (text_mask[sel] > 0).mean() > 0.25:                 # 大部分是字
            continue
        rp = ring_of(filled, x, y, w, h, max(3, W // 250))
        ring_clean = (np.linalg.norm(rp - np.median(rp, axis=0), axis=1) < 30).mean() if len(rp) >= 8 else 0.0
        if ring_clean < 0.6:
            continue                                           # 四周不是干净背景，说明没切完整
        if os.environ.get("PICDBG"):
            print("   blob", (x, y, w, h), "edge", round(float(edge[sel].mean()), 2), "ring", round(float(ring_clean), 2), "text", round(float((text_mask[sel] > 0).mean()), 2))
        filled = cv2.dilate(filled, np.ones((3, 3), np.uint8))
        p = 2
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        alpha = cv2.GaussianBlur(filled[y0:y1, x0:x1], (3, 3), 0)
        parts = [_ramp(edge[sel].mean(), 0.4, 0.7), _ramp(ring_clean, 0.5, 0.9), _ramp((filled[y:y + h, x:x + w] > 0).mean(), 0.6, 0.92)]
        pics.append(Icon(x0, y0, np.dstack([img_bgr[y0:y1, x0:x1], alpha]), "图片", (x, y, w, h),
                         int(round(100 * (0.5 * min(parts) + 0.5 * sum(parts) / 3))),
                         "细节 %d · 独立 %d · 形状 %d" % tuple(round(p * 100) for p in parts)))
        masks.append(sel)
        if ring_clean >= 0.9:
            smooth = np.maximum(smooth, filled)
        erase = np.maximum(erase, cv2.dilate(filled, np.ones((3, 3), np.uint8)))
    # ---- 方法二：补漏。按纹理密度找方法一没抠到的矩形照片（条件卡得严，宁缺毋滥） ----
    for (x, y, w, h), filled in _find_photo_rects(img_bgr, text_mask, edge, erase):
        (sx, sy, sw, sh), sides = _snap_photo_rect(agx, agy, (x, y, w, h))
        if sides < 4 or detail[sy:sy + sh, sx:sx + sw].mean() < 0.6 or min(sw, sh) < PIC_MIN_RATIO * W * 0.85:
            continue                                           # 收不成一个规整的矩形：不可靠，不抠
        x, y, w, h = sx, sy, sw, sh
        filled = _rect_mask(img_bgr, x, y, w, h)
        if (erase[filled > 0] > 0).mean() > 0.3:
            continue
        sel = filled > 0
        p = 2
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        alpha = cv2.GaussianBlur(filled[y0:y1, x0:x1], (3, 3), 0)
        pics.append(Icon(x0, y0, np.dstack([img_bgr[y0:y1, x0:x1], alpha]), "图片", (x, y, w, h), 65, "按纹理补找到的矩形照片"))
        masks.append(sel)
        erase = np.maximum(erase, cv2.dilate(filled, np.ones((3, 3), np.uint8)))
    return pics, erase, masks, smooth


def find_illustrations(img_bgr, text_mask, taken):
    """找"组合插画"：由好几样东西叠在一起画成的一幅图（细菌 + 上升箭头 + 柱子；带坐标轴的面积图）。
    它们没有矩形边界，零件之间还隔着空隙，按"一块一块"找只会抠出几个碎片。
    办法：把彼此靠得很近的内容归成一簇；一簇里如果颜色是渐变、晕染的（而不是几种平涂的颜色），
    又摆在干净的底色上，就整簇当成一张图。平涂的图标排成的流程图颜色种类少，不会被误并。
    返回 [(Icon, 整页布尔掩膜), ...]"""
    H, W = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    kb = (W // 9) | 1
    small = cv2.resize(img_bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    bg_est = cv2.resize(cv2.medianBlur(small, (kb // 2) | 1), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    bg_hsv = cv2.cvtColor(np.clip(bg_est, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    light_bg = (bg_hsv[..., 2] > 215) & (bg_hsv[..., 1] < 45)   # 只在浅色卡片上找
    # "内容" = 不属于大片平坦区域的像素（卡片底色、卡片顶上的浅色条都是大片平坦的，不算）
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    edge0 = (np.sqrt(gx ** 2 + gy ** 2).max(axis=2) / 4.0 > 10).astype(np.uint8)
    flat0 = (cv2.dilate(edge0, np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
    nf, labf, stf, _ = cv2.connectedComponentsWithStats(flat0, connectivity=4)
    bigf = stf[:, cv2.CC_STAT_AREA] >= 0.004 * H * W
    bigf[0] = False
    body = ~bigf[labf] & light_bg & (np.linalg.norm(f - bg_est, axis=2) > 10)
    tm = cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) > 0
    strong = np.linalg.norm(f - bg_est, axis=2) > 45
    body &= ~tm & (cv2.dilate(taken, np.ones((9, 9), np.uint8)) == 0)
    body_raw = body.astype(np.uint8)                           # 连一两个像素宽的细线也留着（坐标轴）
    body = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 归簇时先不看细线（卡片边框、分隔线会把上下几行的东西串成一簇）
    seed = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    g = max(9, W // 70) | 1                                     # 相隔不到这么远的算一簇
    group = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(group, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min(w, h) < 0.055 * W or max(w, h) > 0.35 * W or area < 0.3 * w * h:
            continue
        if x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3:
            continue
        sel_g = lab == i
        px = img_bgr[sel_g & (body > 0)]
        if len(px) < 400:
            continue
        # 平涂的图标只有两三种颜色；插画是渐变、晕染的。量法：去掉最常见的 3 种颜色之后，还剩多少像素
        q = (px >> 3).astype(np.int32)
        key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
        top = np.argsort(np.bincount(key))[::-1][:3]
        pf = px.astype(np.float32)
        near = np.zeros(len(px), bool)
        for t in top:
            c = np.array([(t // 1024) * 8 + 4, (t // 32 % 32) * 8 + 4, (t % 32) * 8 + 4], np.float32)
            near |= np.linalg.norm(pf - c, axis=1) < 30
        shades = int(round(100 * (1 - near.mean())))
        # 四周得是干净的底色
        p = max(6, W // 150)
        X0, Y0, X1, Y1 = max(0, x - 3 * p), max(0, y - 3 * p), min(W, x + w + 3 * p), min(H, y + h + 3 * p)
        sub = sel_g[Y0:Y1, X0:X1].astype(np.uint8)
        ring = (cv2.dilate(sub, np.ones((3 * p + 1, 3 * p + 1), np.uint8)) > 0) & (cv2.dilate(sub, np.ones((p + 1, p + 1), np.uint8)) == 0)
        ring &= ~tm[Y0:Y1, X0:X1]
        rp = f[Y0:Y1, X0:X1][ring]
        ring_clean = float((np.linalg.norm(rp - np.median(rp, axis=0), axis=1) < 22).mean()) if len(rp) >= 30 else 0.0
        if os.environ.get("PICDBG"):
            print("   illus?", (x, y, w, h), "shades", shades, "ring", round(ring_clean, 2), "fill", round(area / float(w * h), 2))
        if shades < 45 or ring_clean < 0.7:
            continue
        sel = (sel_g & (cv2.dilate(body, np.ones((3, 3), np.uint8)) > 0)).astype(np.uint8)
        # 簇边上单独摆着的平涂小件（流程图里指向下一步的箭头）不属于这幅插画：颜色单一、边缘清晰、个头小
        nc, labc, stc, _ = cv2.connectedComponentsWithStats(sel, connectivity=8)
        total = float(sel.sum())
        for j in range(1, nc):
            if stc[j][4] > 0.15 * total or stc[j][4] < 30:
                continue
            piece = labc == j
            if min(stc[j][2], stc[j][3]) <= 9 and max(stc[j][2], stc[j][3]) >= 8 * min(stc[j][2], stc[j][3]):
                sel[piece] = 0                                 # 孤零零的一条细线（卡片的分隔线）
                continue
            pp = f[piece & (strong > 0)]                        # 只看它"实心"的部分，边缘的过渡色不算
            flat_share = float((np.linalg.norm(pp - np.median(pp, axis=0), axis=1) < 30).mean()) if len(pp) >= 20 else 0.0
            sharp = _contour_on_edges(img_bgr, piece.astype(np.uint8) * 255) if flat_share >= 0.45 else 0.0
            # 离这幅画其余部分有多远：柱状图的柱子彼此挨得很近，流程箭头和旁边的图之间隔着明显的空
            others = ((labc > 0) & ~piece).astype(np.uint8)
            px0, py0, pw0, ph0 = [int(v) for v in stc[j][:4]]
            m_ = 3 * g
            wy0, wy1, wx0, wx1 = max(0, py0 - m_), min(H, py0 + ph0 + m_), max(0, px0 - m_), min(W, px0 + pw0 + m_)
            dist = cv2.distanceTransform(1 - others[wy0:wy1, wx0:wx1], cv2.DIST_L2, 3)
            gap = float(dist[piece[wy0:wy1, wx0:wx1]].min())
            if os.environ.get("PICDBG") and stc[j][4] > 150:
                print("      piece", tuple(int(v) for v in stc[j][:4]), "flat", round(flat_share, 2), "sharp", round(sharp, 2), "gap", round(gap, 1))
            if flat_share >= 0.45 and sharp >= 0.93 and gap >= 0.4 * g:
                sel[piece] = 0
        sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # 归簇时没看细线，这里把连在插画上的细线补回来（坐标轴、箭头的尾巴），但不能越出这一簇太远（卡片边框不算）
        thin_all = cv2.dilate(body_raw, np.ones((3, 3), np.uint8))
        nb, labb, stb, _ = cv2.connectedComponentsWithStats(thin_all, connectivity=8)
        for j in np.unique(labb[sel > 0]):
            if j == 0:
                continue
            jx, jy, jw, jh, _ = stb[j]
            if jx >= x - g and jy >= y - g and jx + jw <= x + w + g and jy + jh <= y + h + g:
                sel[labb == j] = 1
        # 渐变淡出的部分（柱子的底部）和底色差得太少会被漏掉，在轮廓上留下缺口；把窄缺口补平
        sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, np.ones((1, 4 * g + 1), np.uint8))   # 同一行上两边都有内容的缺口
        sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, np.ones((4 * g + 1, 1), np.uint8))
        sel[cv2.dilate(taken, np.ones((5, 5), np.uint8)) > 0] = 0
        sel[tm & ~cv2.dilate(sel_g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 0
        cs, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sel, cs, -1, 1, cv2.FILLED)            # 轮廓里面的小洞填上，免得抠出来的图中间缺一块
        sel = cv2.dilate(sel, np.ones((3, 3), np.uint8)) > 0
        if sel.sum() < 400:
            continue
        if (text_mask[sel] > 0).mean() > 0.3:
            continue
        ys, xs = np.where(sel)
        bx, by, bw, bh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        ax0, ay0, ax1, ay1 = max(0, bx - 2), max(0, by - 2), min(W, bx + bw + 2), min(H, by + bh + 2)
        solid = sel[ay0:ay1, ax0:ax1].astype(np.uint8) * 255
        alpha = np.maximum(cv2.GaussianBlur(solid, (5, 5), 0), solid)   # 细线（坐标轴）不能被羽化淡掉
        parts = [_ramp(shades, 35, 65), _ramp(ring_clean, 0.6, 0.92)]
        score = int(round(100 * (0.5 * min(parts) + 0.5 * sum(parts) / 2)))
        ic = Icon(ax0, ay0, np.dstack([img_bgr[ay0:ay1, ax0:ax1], alpha]), "图片", (bx, by, bw, bh), score,
                  "组合插画 · 色彩层次 %d · 独立 %d" % tuple(round(v * 100) for v in parts))
        out.append((ic, sel))
    return out


def find_leftover_pictures(img_bgr, text_mask, taken):
    """第三轮配图识别。前两种方法要求照片有清晰的矩形边界或很强的细节，
    浅色的设备图、小折线图这类"配图"会漏掉。这里换个角度：把文字和已经抠走的东西都去掉以后，
    页面上还有哪些地方"细节成片"？那就是还没抠的配图。
    抠法：以它四周的底色为准，和底色不同的像素连成的整块就是配图的轮廓。
    返回 [(Icon, 整页布尔掩膜), ...]"""
    H, W = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.sqrt(gx ** 2 + gy ** 2).max(axis=2) / 4.0 > 10).astype(np.uint8)
    edge[cv2.dilate(text_mask, np.ones((7, 7), np.uint8)) > 0] = 0
    edge[cv2.dilate(taken, np.ones((9, 9), np.uint8)) > 0] = 0
    k = (W // 90) | 1
    dens = cv2.blur(edge.astype(np.float32), (k, k))
    core = (dens > 0.24).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((k // 2 | 1, k // 2 | 1), np.uint8))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        dbg = os.environ.get("PICDBG") and min(w, h) >= 0.025 * W
        if dbg:
            print("   core", (x, y, w, h), "fill", round(area / float(w * h), 2))
        if min(w, h) < 0.035 * W or max(w, h) > 0.35 * W or area < 0.5 * w * h:
            continue
        if x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3:
            continue
        # 四周一圈的底色
        p = max(6, W // 150)
        X0, Y0, X1, Y1 = max(0, x - 3 * p), max(0, y - 3 * p), min(W, x + w + 3 * p), min(H, y + h + 3 * p)
        box = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        box[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = 1
        ring = (cv2.dilate(box, np.ones((3 * p + 1, 3 * p + 1), np.uint8)) > 0) & (cv2.dilate(box, np.ones((p + 1, p + 1), np.uint8)) == 0)
        ring &= text_mask[Y0:Y1, X0:X1] == 0
        rp = f[Y0:Y1, X0:X1][ring]
        if len(rp) < 30:
            continue
        bg = np.median(rp, axis=0)
        ring_clean = float((np.linalg.norm(rp - bg, axis=1) < 22).mean())
        if dbg:
            print("      ring_clean", round(ring_clean, 2))
        if ring_clean < 0.6:
            continue                                           # 四周不是干净底色：它和别的东西连着，切不干净
        # 和底色不同的像素 → 配图的轮廓（只取和核心连着的那一块）
        diff = (np.linalg.norm(f[Y0:Y1, X0:X1] - bg, axis=2) > 22).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m2, lab2 = cv2.connectedComponents(diff, connectivity=8)
        ids = np.unique(lab2[(lab[Y0:Y1, X0:X1] == i) & (diff > 0)])
        ids = ids[ids > 0]
        if not len(ids):
            continue
        shape = np.isin(lab2, ids).astype(np.uint8)
        cs, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(shape)
        cv2.drawContours(filled, cs, -1, 255, cv2.FILLED)
        bx, by, bw, bh = cv2.boundingRect(filled)
        if dbg:
            print("      shape box", (bx, by, bw, bh), "of", filled.shape[::-1], "fill", round(float((filled[by:by + bh, bx:bx + bw] > 0).mean()), 2))
        if bx <= 1 or by <= 1 or bx + bw >= filled.shape[1] - 1 or by + bh >= filled.shape[0] - 1:
            continue                                           # 轮廓一直延伸到取样范围之外：它是更大东西的一部分
        if min(bw, bh) < 0.035 * W or (filled[by:by + bh, bx:bx + bw] > 0).mean() < 0.45:
            continue
        sel = np.zeros((H, W), bool)
        sel[Y0:Y1, X0:X1] = filled > 0
        if (text_mask[sel] > 0).mean() > 0.3 or (taken[sel] > 0).mean() > 0.2:
            continue
        detail = float(dens[sel].mean())
        parts = [_ramp(detail, 0.1, 0.26), _ramp(ring_clean, 0.6, 0.92), _ramp((filled[by:by + bh, bx:bx + bw] > 0).mean(), 0.45, 0.8)]
        score = int(round(100 * (0.5 * min(parts) + 0.5 * sum(parts) / 3)))
        if os.environ.get("PICDBG"):
            print("   leftover?", (X0 + bx, Y0 + by, bw, bh), score, [round(v, 2) for v in parts])
        if score < 50:
            continue
        ax0, ay0 = X0 + max(0, bx - 2), Y0 + max(0, by - 2)
        ax1, ay1 = min(W, X0 + bx + bw + 2), min(H, Y0 + by + bh + 2)
        alpha = cv2.GaussianBlur(cv2.dilate(sel[ay0:ay1, ax0:ax1].astype(np.uint8) * 255, np.ones((3, 3), np.uint8)), (3, 3), 0)
        ic = Icon(ax0, ay0, np.dstack([img_bgr[ay0:ay1, ax0:ax1], alpha]), "图片", (X0 + bx, Y0 + by, bw, bh), score,
                  "补找到的配图 · 细节 %d · 独立 %d · 形状 %d" % tuple(round(v * 100) for v in parts))
        out.append((ic, sel))
    return out


def _find_photo_rects(img_bgr, text_mask, edge, taken):
    """照片的特点是"到处都有细节"，而它外面一圈（底板、留白）是平的。
    做法：先按边缘密度找到照片的"核心"，再从核心的四边分别往外走，
    走到"外面开始变平、并且交界处是一条笔直强边"的位置，就是照片的真实边界。
    返回 [((x, y, w, h), 整页掩膜), ...]"""
    H, W = img_bgr.shape[:2]
    e = edge.copy()
    e[cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) > 0] = 0
    k = (W // 60) | 1
    dens = cv2.blur(e.astype(np.float32), (k, k))
    core = (dens > 0.33).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((k // 2 | 1, k // 2 | 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    ef = e.astype(np.float32)
    limit = int(0.2 * W)

    def walk(start, step, dens_prof, line_prof, thr, core_len):
        """从核心的边缘 start 沿 step 方向往外走，返回 (边界位置, 该处直线强度)。
        先走到"细节密度明显掉下来"的位置；那里如果正好有一条笔直的强边，就是照片边界。
        没有的话（比如照片上方是一片天空），再往外找一小段，找不到就退回来。"""
        n_ = len(dens_prof)
        p = start
        while 2 <= p + step < n_ - 2 and dens_prof[p + step] >= thr and abs(p - start) < limit:
            p += step
        def line_near(c, half=5):
            lo, hi = max(1, c - half), min(n_ - 2, c + half)
            q = lo + int(np.argmax(line_prof[lo:hi + 1]))
            return q, float(line_prof[q])
        q, strength = line_near(p)
        if strength >= 12:
            return q, strength
        gap = int(max(15, 0.5 * core_len))
        for d in range(6, gap):
            c = p + step * d
            if not 8 <= c < n_ - 8:
                break
            if line_prof[c] >= 12:
                beyond = dens_prof[c + step * 2:c + step * 8:step] if step > 0 else dens_prof[c - 7:c - 1]
                if len(beyond) and beyond.max() < thr:
                    return c, float(line_prof[c])
        return q, strength

    rects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min(w, h) < 0.02 * W or area < 0.4 * w * h:
            continue
        box = [x, y, x + w - 1, y + h - 1]
        strengths = [0, 0, 0, 0]
        for _ in range(2):                                     # 走两遍：第一遍定出大致范围，第二遍用更准的范围重新量
            bx0, by0, bx1, by1 = box
            sx0, sx1 = bx0 + (bx1 - bx0) // 6, bx1 - (bx1 - bx0) // 6 + 1
            sy0, sy1 = by0 + (by1 - by0) // 6, by1 - (by1 - by0) // 6 + 1
            if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                break
            row_d = cv2.blur(ef[:, sx0:sx1].mean(axis=1).reshape(-1, 1), (1, 7)).ravel()
            col_d = cv2.blur(ef[sy0:sy1, :].mean(axis=0).reshape(-1, 1), (1, 7)).ravel()
            row_line, col_line = gy[:, sx0:sx1].mean(axis=1), gx[sy0:sy1, :].mean(axis=0)
            rthr = 0.3 * float(row_d[y:y + h].mean())
            cthr = 0.3 * float(col_d[x:x + w].mean())
            top, s0 = walk(y, -1, row_d, row_line, rthr, h)
            bot, s1 = walk(y + h - 1, 1, row_d, row_line, rthr, h)
            left, s2 = walk(x, -1, col_d, col_line, cthr, w)
            right, s3 = walk(x + w - 1, 1, col_d, col_line, cthr, w)
            box, strengths = [left, top, right, bot], [s0, s1, s2, s3]
        left, top, right, bot = box
        rw, rh = right - left + 1, bot - top + 1
        if min(rw, rh) < PIC_MIN_RATIO * W or max(rw, rh) > 0.6 * W:
            continue
        if left <= 2 or top <= 2 or right >= W - 3 or bot >= H - 3:
            continue                                           # 贴着画面边缘的不抠
        if sum(v >= 12 for v in strengths) < 3:
            continue                                           # 没有清晰的矩形边界
        if dens[top:bot + 1, left:right + 1].mean() < 0.4 or core[top:bot + 1, left:right + 1].mean() < 0.58:
            continue                                           # 整体细节不够多，不像照片
        if min(rw, rh) < 0.065 * W:
            continue
        if (text_mask[top:bot + 1, left:right + 1] > 0).mean() > 0.2:
            continue
        if (taken[top:bot + 1, left:right + 1] > 0).mean() > 0.3:
            continue
        def overlap_small(a, b_):
            iw = max(0, min(a[0] + a[2], b_[0] + b_[2]) - max(a[0], b_[0]))
            ih = max(0, min(a[1] + a[3], b_[1] + b_[3]) - max(a[1], b_[1]))
            return iw * ih / float(min(a[2] * a[3], b_[2] * b_[3]))
        if any(overlap_small((left, top, rw, rh), r_) > 0.4 for r_ in rects):
            continue                                           # 同一张照片的几块核心会得到差不多的矩形，留先找到的
        if os.environ.get("PICDBG"):
            print("   rect", (left, top, rw, rh), "strength", [round(v) for v in strengths], "dens", round(float(dens[top:bot + 1, left:right + 1].mean()), 2),
                  "core", round(float(core[top:bot + 1, left:right + 1].mean()), 2))
        rects.append((left, top, rw, rh))

    return [((x, y, w, h), _rect_mask(img_bgr, x, y, w, h)) for x, y, w, h in rects]


def _rect_mask(img_bgr, x, y, w, h):
    """矩形照片的掩膜：一个矩形，四个圆角挖掉。
    圆角的判断：四个角各看一小块，和角外侧颜色相同、并且和角尖连通的像素不属于照片。"""
    H, W = img_bgr.shape[:2]
    filled = np.zeros((H, W), np.uint8)
    filled[y:y + h, x:x + w] = 255
    rr = int(min(w, h) * 0.14)
    for cx, cy, ox, oy in ((x, y, -3, -3), (x + w - 1, y, 3, -3), (x, y + h - 1, -3, 3), (x + w - 1, y + h - 1, 3, 3)):
        sx, sy = min(W - 1, max(0, cx + ox)), min(H - 1, max(0, cy + oy))
        bgc = img_bgr[sy, sx].astype(np.float32)
        ax0, ax1 = (cx, cx + rr) if ox < 0 else (cx - rr + 1, cx + 1)
        ay0, ay1 = (cy, cy + rr) if oy < 0 else (cy - rr + 1, cy + 1)
        patch = img_bgr[ay0:ay1, ax0:ax1].astype(np.float32)
        if patch.size == 0:
            continue
        same = (np.linalg.norm(patch - bgc, axis=2) < 22).astype(np.uint8)
        m2, lab2 = cv2.connectedComponents(same, connectivity=4)
        tip = lab2[0 if oy < 0 else -1, 0 if ox < 0 else -1]
        if tip > 0:
            cut = lab2 == tip
            if cut.mean() < 0.45:                              # 真圆角最多挖掉这一小块的约两成；挖太多说明判断错了
                filled[ay0:ay1, ax0:ax1][cut] = 0
    return filled


def _snap_photo_rect(agx, agy, box):
    """把一个大致的框收紧成矩形照片的真实边界。agx / agy 是灰度图横向、纵向的梯度绝对值。
    矩形照片的四条边各是一条"又长又直"的明暗分界线。四条边分别从框外一点点往里找，
    找到第一条贯穿大半个边长的分界线，就是那条边。找不到的边保持不动。
    返回 ((x, y, w, h), 找到了几条边)。"""
    H, W = agx.shape
    x, y, w, h = box
    out_pad = 6

    def first_line(profile_of, start, step, limit):
        """profile_of(p) 给出位置 p 上"是分界线"的比例；从 start 沿 step 走，返回第一个 ≥ 0.62 的位置。"""
        p = start
        for _ in range(limit):
            if profile_of(p) >= 0.62:
                # 一条边通常占 2–3 个像素，取其中最强的那一行 / 列
                best = max(range(p, p + 3 * step, step), key=profile_of)
                return best
            p += step
        return None

    xa, xb = x + int(0.12 * w), x + w - int(0.12 * w)           # 量横线时只看中间这段（避开圆角）
    ya, yb = y + int(0.12 * h), y + h - int(0.12 * h)
    if xb - xa < 10 or yb - ya < 10:
        return box, 0
    row = lambda p: float((agy[p, xa:xb] > 14).mean()) if 1 <= p < H - 1 else 0.0
    col = lambda p: float((agx[ya:yb, p] > 14).mean()) if 1 <= p < W - 1 else 0.0
    top = first_line(row, y - out_pad, 1, out_pad + int(0.25 * h))
    bot = first_line(row, y + h - 1 + out_pad, -1, out_pad + int(0.25 * h))
    left = first_line(col, x - out_pad, 1, out_pad + int(0.25 * w))
    right = first_line(col, x + w - 1 + out_pad, -1, out_pad + int(0.25 * w))
    found = sum(v is not None for v in (top, bot, left, right))
    top = y if top is None else top
    bot = y + h - 1 if bot is None else bot
    left = x if left is None else left
    right = x + w - 1 if right is None else right
    return (left, top, right - left + 1, bot - top + 1), found


def _contour_on_edges(img_bgr, filled, thr=14):
    """真正独立的图形，轮廓处处都有明显的颜色变化；从渐变色带上"切"下来的一块，切口那一侧没有。
    返回轮廓上有明显边缘的比例。"""
    x, y, w, h = cv2.boundingRect(filled)
    p = 4
    H, W = filled.shape
    x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    sub = filled[y0:y1, x0:x1]
    f = img_bgr[y0:y1, x0:x1].astype(np.float32)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    strong = (np.sqrt(gx ** 2 + gy ** 2).max(axis=2) / 4.0 > thr).astype(np.uint8)
    strong = cv2.dilate(strong, np.ones((7, 7), np.uint8))
    contour = (sub > 0) & (cv2.erode(sub, np.ones((3, 3), np.uint8)) == 0)
    return float(strong[contour].mean()) if contour.any() else 0.0


def _flat_icon_rgba(crop_bgr, filled, ring_px):
    """单色图标（白色线条图标、纯色箭头……）的精细抠图。
    直接按掩膜切下来，边缘一圈会带着背景色（白图标带一圈绿边）。这里改成"颜色分离"：
    每个像素 = 图标色 × 透明度 + 背景色 × (1 − 透明度)，解出透明度，再把颜色统一设成图标色。
    这样边缘是半透明的纯图标色，放到任何背景上都干净。不是单色图标就返回 None，按原样处理。"""
    if len(ring_px) < 8:
        return None
    bg = np.median(ring_px, axis=0)
    if (np.linalg.norm(ring_px - bg, axis=1) < 28).mean() < 0.85:
        return None                                            # 背景不是一种颜色，解不准
    f = crop_bgr.astype(np.float32)
    m = filled > 0
    dist = np.linalg.norm(f - bg, axis=2)
    core = m & (dist > 0.7 * np.percentile(dist[m], 90))
    if core.sum() < 10:
        return None
    fg = np.median(f[core], axis=0)
    if np.linalg.norm(fg - bg) < 60 or (np.linalg.norm(f[core] - fg, axis=1) < 40).mean() < 0.9:
        return None                                            # 图标本身不止一种颜色
    v = fg - bg
    a = np.clip(((f - bg) @ v) / float(v @ v), 0, 1)
    off = np.linalg.norm(f - (bg + a[..., None] * v), axis=2)  # 不在"背景色—图标色"连线上的像素：说明里面还有别的颜色
    if (off[m] > 45).mean() > 0.06:
        return None
    near = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = np.where(near, a, 0)
    a[a < 0.08] = 0
    out = np.zeros(crop_bgr.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(fg, 0, 255).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def _ramp(v, lo, hi):
    """把 v 从 [lo, hi] 线性映射到 [0, 1]。"""
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def score_icon(img_bgr, filled, text_mask, ring_px):
    """给一个图标候选逐项打分，返回 (总分 0–100, 分项说明)。四项各回答一个问题：
      独立  它四周是不是一片干净的底色？（照片里的一块颜色不是）
      轮廓  它的边界是不是处处都有清晰的颜色变化？（从渐变色带上切下来的一块不是）
      完整  它是不是某个更大色块的一角？（四周有和它边缘同色的东西就是）
      内容  它里面有没有图案？（纯色的圆、矩形、胶囊只是形状，不是图标）
    总分看最弱的一项为主：任何一项很差，这个候选就不可信。"""
    x, y, w, h = cv2.boundingRect(filled)
    sub = filled[y:y + h, x:x + w] > 0
    roi = img_bgr[y:y + h, x:x + w].astype(np.float32)
    free = sub & (text_mask[y:y + h, x:x + w] == 0)
    if free.sum() < 12 or len(ring_px) < 8:
        return 0, "太小"
    alone = _ramp((np.linalg.norm(ring_px - np.median(ring_px, axis=0), axis=1) < 30).mean(), 0.45, 0.9)
    outline = _ramp(_contour_on_edges(img_bgr, filled), 0.86, 0.985)
    whole = 1.0
    band = free & (cv2.erode(sub.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0)
    if band.sum() >= 12:
        own = np.median(roi[band], axis=0)
        if (np.linalg.norm(roi[band] - own, axis=1) < 25).mean() > 0.6:
            whole = 1.0 - _ramp((np.linalg.norm(ring_px - own, axis=1) < 14).mean(), 0.06, 0.25)
    content = 1.0
    inner = cv2.erode(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    inner &= cv2.dilate(text_mask[y:y + h, x:x + w], np.ones((5, 5), np.uint8)) == 0
    cs, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = sub.sum() / max(1.0, cv2.contourArea(cv2.convexHull(np.vstack(cs))))
    # 只有"基本形状"才按纯色扣分：填满外接矩形（矩形、胶囊）或者正好是个椭圆。箭头、对勾这类不算
    extent = sub.mean()
    basic = extent > 0.86 or (solidity > 0.93 and 0.72 < extent < 0.84)
    if basic and inner.sum() < 30:
        content = 0.2                                          # 很小的实心方块 / 圆点：是形状之间的空隙或项目符号，不是图标
    elif basic:
        g = cv2.cvtColor(img_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        content = _ramp((np.sqrt(gx ** 2 + gy ** 2)[inner] > 40).mean(), 0.01, 0.07)
    parts = [alone, outline, whole, content]
    total = int(round(100 * (0.65 * min(parts) + 0.35 * sum(parts) / 4)))
    why = "独立 %d · 轮廓 %d · 完整 %d · 内容 %d" % tuple(round(p * 100) for p in parts)
    return total, why


def _is_piece_or_plain(img_bgr, filled, text_mask, ring_px):
    """两种不该当成图标的情况：
      ① 它只是某个更大色块（标题色带、卡片）被切下来的一角 —— 四周有不少像素和它自己的边缘同色
      ② 它是一块没有内容的纯色形状（标签底、表格格子）—— 内部没有线条，轮廓又很规整
    """
    x, y, w, h = cv2.boundingRect(filled)
    sub = filled[y:y + h, x:x + w] > 0
    roi = img_bgr[y:y + h, x:x + w].astype(np.float32)
    free = sub & (text_mask[y:y + h, x:x + w] == 0)
    if free.sum() < 12:
        return True
    # ① 取紧贴轮廓内侧的一圈像素，看四周有多少和它同色
    band = free & (cv2.erode(sub.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0)
    if band.sum() >= 12:
        own = np.median(roi[band], axis=0)
        if (np.linalg.norm(roi[band] - own, axis=1) < 25).mean() > 0.6:          # 边缘本身是一种颜色才有意义
            if (np.linalg.norm(ring_px - own, axis=1) < 14).mean() > 0.18:
                return True
    # ② 内部（去掉边缘和文字）几乎没有线条，且形状接近凸的（矩形、圆、胶囊）
    inner = cv2.erode(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    inner &= cv2.dilate(text_mask[y:y + h, x:x + w], np.ones((5, 5), np.uint8)) == 0
    if inner.sum() >= 30:
        g = cv2.cvtColor(img_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        busy = (np.sqrt(gx ** 2 + gy ** 2)[inner] > 40).mean()
        cs, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(cs))
        solidity = sub.sum() / max(1.0, cv2.contourArea(hull))
        if busy < 0.03 and solidity > 0.9:
            return True
    return False


def _merge_close(masks, gap, max_side):
    boxes = [cv2.boundingRect(m.astype(np.uint8)) for m in masks]
    merged = True
    while merged:
        merged = False
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                dy = max(0, max(ay, by) - min(ay + ah, by + bh))
                ux, uy = min(ax, bx), min(ay, by)
                uw, uh = max(ax + aw, bx + bw) - ux, max(ay + ah, by + bh) - uy
                if dx <= gap and dy <= gap and max(uw, uh) <= max_side:
                    masks[i] = masks[i] | masks[j]
                    boxes[i] = (ux, uy, uw, uh)
                    del masks[j], boxes[j]
                    merged = True
                    break
            if merged:
                break
    return masks


# ----------------------------------------------------------------------------
# 把相邻的行合并成一个文本框（同一段落 / 同一组要点）
# ----------------------------------------------------------------------------
def group_lines(lines):
    lines = sorted(lines, key=lambda l: (l.y, l.x))
    blocks = []
    for ln in lines:
        target = None
        for b in blocks:
            last = b.lines[-1]
            same_size = abs(last.size_px - ln.size_px) < 0.15 * max(last.size_px, ln.size_px)
            same_style = last.bold == ln.bold and _color_close(last.color, ln.color)
            gap = ln.baseline - last.baseline
            close = 0.9 * ln.size_px < gap < 2.2 * ln.size_px
            left = abs(last.ink[0] - ln.ink[0]) < 0.6 * ln.size_px
            center = abs((last.ink[0] + last.ink[2]) - (ln.ink[0] + ln.ink[2])) / 2 < 0.6 * ln.size_px
            if same_size and same_style and close and (left or center) and not ln.badge and not last.badge:
                target = b
                if not left and center:
                    b.align = "center"
                break
        if target:
            target.lines.append(ln)
        else:
            blocks.append(Block([ln]))
    return blocks


def _color_close(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) < 90


# ----------------------------------------------------------------------------
# 第 4 步：写 PPTX
# ----------------------------------------------------------------------------
def add_slide(prs, clean_bgr, blocks, icons, img_w, img_h, font_name):
    slide = prs.slides.add_slide(prs.slide_layouts[6])        # 空白版式
    emu_per_px = SLIDE_W_EMU / img_w

    ok, buf = cv2.imencode(".png", clean_bgr)
    slide.shapes.add_picture(io.BytesIO(buf.tobytes()), 0, 0, prs.slide_width, prs.slide_height)

    count = {}
    for ic in sorted(icons, key=lambda i: i.kind != "图片"):    # 先放图片、再放图标：都是独立元素，可单独移动 / 缩放
        ok, buf = cv2.imencode(".png", ic.rgba)
        h, w = ic.rgba.shape[:2]
        pic = slide.shapes.add_picture(io.BytesIO(buf.tobytes()), Emu(int(ic.x * emu_per_px)), Emu(int(ic.y * emu_per_px)),
                                       Emu(int(w * emu_per_px)), Emu(int(h * emu_per_px)))
        count[ic.kind] = count.get(ic.kind, 0) + 1
        pic.name = f"{ic.kind} {count[ic.kind]}"

    for b in blocks:
        first = b.lines[0]
        sizes = [l.size_px for l in b.lines]
        # 同一个文本框里的几行字号差不多时取最小的：有的行为了不顶到旁边的东西调小过，整框跟着它走才整齐
        size_px = float(min(sizes)) if max(sizes) <= 1.16 * min(sizes) else float(np.median(sizes))
        size_pt = max(6.0, round(size_px * emu_per_px / EMU_PER_PT * 2) / 2)     # 取到 0.5 磅
        if len(b.lines) > 1:                                   # 多行：用实际行间距
            pitch_px = (b.lines[-1].baseline - first.baseline) / (len(b.lines) - 1)
        else:
            pitch_px = size_px * LINE_SPACING
        pitch_pt = pitch_px * emu_per_px / EMU_PER_PT

        left_px = min(l.ink[0] for l in b.lines)
        right_px = max(l.ink[2] for l in b.lines)
        top_px = first.baseline - BASELINE_RATIO * pitch_px
        slack = size_px * 0.6                                  # 右边留点余量，换了字体也不容易折行
        if first.badge and first.disc:                         # 序号：文本框以圆心为中心、居中对齐，换字体也不会偏
            b.align = "center"
            left_px, right_px = first.disc[0] - 2 * first.disc[2], first.disc[0] + 2 * first.disc[2]
        elif b.align == "center":
            left_px -= slack
            right_px += slack
        else:
            right_px += 2 * slack

        box = slide.shapes.add_textbox(
            Emu(int(left_px * emu_per_px)), Emu(int(top_px * emu_per_px)),
            Emu(int((right_px - left_px) * emu_per_px)),
            Emu(int(pitch_px * len(b.lines) * emu_per_px)))
        tf = box.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.TOP

        for i, ln in enumerate(b.lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER if b.align == "center" else PP_ALIGN.LEFT
            p.line_spacing = Pt(pitch_pt)
            runs = ln.runs or [(ln.text, ln.color, ln.bold, ln.size_px)]
            for seg, color, bold, seg_px in runs:
                run = p.add_run()
                run.text = seg
                # 单色行用整个文本框统一的字号；多色行每段用自己的字号
                run_pt = size_pt if len(runs) == 1 else max(6.0, round(seg_px * emu_per_px / EMU_PER_PT * 2) / 2)
                run.font.size = Pt(run_pt)
                if ln.italic:
                    run.font.italic = True
                if abs(ln.fit) >= 0.004 and not ln.badge:      # 字距：让这一行和原图一样宽，不会顶到旁边的东西
                    run._r.get_or_add_rPr().set("spc", str(int(round(ln.fit * run_pt * 100))))
                run.font.bold = bold
                run.font.color.rgb = RGBColor(*color)
                run.font.name = font_name
                rPr = run._r.get_or_add_rPr()                  # 中文要单独指定东亚字体
                ea = rPr.find(qn("a:ea"))
                if ea is None:
                    ea = rPr.makeelement(qn("a:ea"), {})
                    rPr.append(ea)
                ea.set("typeface", font_name)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 序号：圆圈 + 里面的一个数字或字母（①②③、Ⓐ Ⓑ Ⓒ）
# 这类东西最容易"图标和字搅在一起"，所以单独走一条路，所有序号用同一种处理方式：
#     圆圈 → 一个独立的图形（把里面的字抹掉，只留圆）
#     里面的字 → 一个文本框，叠在圆圈上面
# ----------------------------------------------------------------------------
@dataclass
class Badge:
    cx: float
    cy: float
    r: float
    fill: np.ndarray            # 圆的主色 (B, G, R)
    glyph: np.ndarray           # 整页大小的布尔掩膜：圆里面"不是主色"的像素，也就是那个字
    text: str = ""
    conf: float = 0.0
    rim: float = 0.0            # 圆周内侧一圈有多少是主色（越接近 1 越像一个完整的圆）


def find_discs(img_bgr):
    """找出页面上所有"实心圆 + 里面有东西"的地方。里面是字还是图案，这一步不管。"""
    H, W = img_bgr.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 5)
    rmin, rmax = int(0.008 * W), int(0.03 * W)
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=rmin * 2, param1=120, param2=22,
                             minRadius=rmin, maxRadius=rmax)
    discs = []
    if found is None:
        return discs
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    for cx, cy, r in found[0]:
        cx, cy, r = float(cx), float(cy), float(r)
        x0, y0 = int(max(0, cx - r * 1.5)), int(max(0, cy - r * 1.5))
        x1, y1 = int(min(W, cx + r * 1.5 + 1)), int(min(H, cy + r * 1.5 + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        roi = img_bgr[y0:y1, x0:x1].astype(np.float32)
        inside, outside = d < r * 0.88, (d > r * 1.12) & (d < r * 1.4)
        if inside.sum() < 30 or outside.sum() < 30:
            continue
        _, lab, cen = cv2.kmeans(roi[inside], 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        fill = cen[int(np.bincount(lab.ravel(), minlength=2).argmax())]
        near = np.linalg.norm(roi - fill, axis=2) < 32
        share = near[inside].mean()                            # 圆内主色占比
        rim = near[(d > r * 0.72) & (d < r * 0.9)].mean()      # 圆周内侧一圈：应该几乎全是主色
        leak = near[outside].mean()                            # 圆外一圈：应该几乎没有主色
        glyph_roi = inside & (np.linalg.norm(roi - fill, axis=2) >= 45)
        gfrac = glyph_roi.sum() / float(inside.sum())
        if not (rim > 0.85 and leak < 0.25 and 0.5 < share < 0.97 and 0.04 < gfrac < 0.45):
            continue
        glyph = np.zeros((H, W), bool)
        glyph[y0:y1, x0:x1] = glyph_roi
        d_ = Badge(cx, cy, r, fill, glyph, rim=float(rim))
        d_.leak = float(leak)
        discs.append(d_)
    return discs


_SERIAL_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _read_glyph(glyph_crop):
    """把圆圈里的那个字单独拿出来认一遍。输入是布尔掩膜（True = 笔画），统一成白底黑字再识别，
    这样深底白字、白底绿字都是同一种输入。返回 (字, 置信度)。"""
    h, w = glyph_crop.shape
    scale = 64.0 / max(1, h)
    big = cv2.resize(glyph_crop.astype(np.uint8) * 255, (max(8, int(w * scale)), 64), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((112, big.shape[1] + 96), 255, np.uint8)
    canvas[24:88, 48:48 + big.shape[1]] = 255 - big
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    try:                                                       # 首选：RapidOCR 只跑"认字"这一步（不用再找文字位置）
        from rapidocr_onnxruntime import RapidOCR
        if not hasattr(ocr_rapid, "engine"):
            ocr_rapid.engine = RapidOCR()
        res = ocr_rapid.engine(bgr, use_det=False, use_cls=False, use_rec=True)[0]
        if res:
            item = res[0]
            text = next((v for v in item if isinstance(v, str)), "")
            conf = next((float(v) for v in item if isinstance(v, (float, np.floating))), 0.0)
            return text.strip(), conf
        return "", 0.0
    except ImportError:
        pass
    except Exception:
        return "", 0.0
    try:                                                       # 备用：tesseract 的"单个字符"模式
        ok, buf = cv2.imencode(".png", bgr)
        out = subprocess.run(["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "10", "-c",
                              "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", "tsv"],
                             input=buf.tobytes(), capture_output=True, check=True).stdout.decode("utf-8", "ignore")
        best = ("", 0.0)
        for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
            t = (r.get("text") or "").strip()
            if t and float(r["conf"]) / 100.0 > best[1]:
                best = (t, float(r["conf"]) / 100.0)
        return best
    except Exception:
        return "", 0.0


def find_badges(img_bgr, lines):
    """从圆里面挑出"序号"：里面是一两个数字 / 字母的才算。里面是图案的圆（圆形图标）不算，交给图标流程。
    返回 (序号列表, 更新后的文字行列表)。每个序号里的字都会成为一行独立的文字。"""
    H, W = img_bgr.shape[:2]
    badges, used = [], set()
    find_badges.round_icons = []                               # 里面是图案而不是字的圆：圆形图标，整个圆连图案一起抠
    for b in find_discs(img_bgr):
        ys, xs = np.where(b.glyph)
        if len(ys) < 6:
            continue
        if any(len(ln.text.strip()) > 2 and ln.x <= b.cx <= ln.x + ln.w and ln.y <= b.cy <= ln.y + ln.h for ln in lines):
            continue                                           # 落在一行文字里面：那是字母"o""e"这类字形里的圆，不是图标
        b.is_serial = False
        find_badges.round_icons.append(b)
        gx0, gy0, gx1, gy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        gh = gy1 - gy0
        if not 0.3 * 2 * b.r <= gh <= 0.85 * 2 * b.r:
            continue                                           # 字的高度应该占圆的三成到八成
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b.glyph[gy0:gy1, gx0:gx1].astype(np.uint8), connectivity=8)
        if sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.04 * len(ys)) > 2:
            continue                                           # 零件太多：是图案不是字
        # 先看 OCR 有没有在这个圆里认出过字；没有的话把圆里的笔画单独拿去认
        inside = [i for i, ln in enumerate(lines)
                  if np.hypot(ln.x + ln.w / 2 - b.cx, ln.y + ln.h / 2 - b.cy) < 0.7 * b.r]
        text, conf = "", 0.0
        for i in inside:
            t = lines[i].text.strip()
            if 1 <= len(t) <= 2 and all(c in _SERIAL_CHARS for c in t):
                text, conf = t, float(lines[i].score)
        if not text:
            text, conf = _read_glyph(b.glyph[gy0:gy1, gx0:gx1])
        if not (1 <= len(text) <= 2 and all(c in _SERIAL_CHARS for c in text) and conf >= 0.4):
            continue
        b.text, b.conf, b.is_serial = text, conf, True
        badges.append(b)
        used.update(inside)
    # 圆形图标的条件卡严一点：够大，并且圆外一圈完全没有同色的东西（定位针那种"圆 + 尾巴"的不算）
    find_badges.round_icons = [b for b in find_badges.round_icons
                               if not b.is_serial and b.r >= 0.0105 * W and b.leak < 0.05]
    out = [ln for i, ln in enumerate(lines) if i not in used]
    for b in badges:                                           # 每个序号的字：一行独立的文字，位置就是笔画的包围盒
        ys, xs = np.where(b.glyph)
        ln = Line(b.text, int(xs.min()), int(ys.min()), int(xs.max() + 1 - xs.min()), int(ys.max() + 1 - ys.min()))
        ln.score, ln.badge, ln.disc = b.conf, True, (b.cx, b.cy, b.r)
        out.append(ln)
    return badges, out


def badge_icons(img_bgr, badges):
    """把每个序号的圆圈做成独立图形：字的位置用圆的颜色补上，只留一个干净的圆。
    返回 (图形列表, 要从底图抹掉的掩膜, 图标检测要避开的区域)。"""
    H, W = img_bgr.shape[:2]
    icons, erase, avoid = [], np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    for b in badges:
        r = int(np.ceil(b.r)) + 3
        x0, y0, x1, y1 = max(0, int(b.cx) - r), max(0, int(b.cy) - r), min(W, int(b.cx) + r + 1), min(H, int(b.cy) + r + 1)
        crop = img_bgr[y0:y1, x0:x1].copy()
        g = cv2.dilate(b.glyph[y0:y1, x0:x1].astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        crop = cv2.inpaint(crop, g, 3, cv2.INPAINT_TELEA)       # 用圆自己的颜色把字补掉（保留圆的渐变和边缘）
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (np.hypot(xx - b.cx, yy - b.cy) <= b.r + 1.5).astype(np.uint8) * 255
        alpha = cv2.GaussianBlur(disc, (3, 3), 0)
        bw = int(2 * b.r + 3)
        icons.append(Icon(x0, y0, np.dstack([crop, alpha]), "图标", (int(b.cx - b.r - 1), int(b.cy - b.r - 1), bw, bw),
                          int(round(100 * min(1.0, b.rim))), f"序号圆圈 · 里面的“{b.text}”是单独的文字"))
        cv2.circle(erase, (int(round(b.cx)), int(round(b.cy))), int(b.r + 3), 255, -1)
        cv2.circle(avoid, (int(round(b.cx)), int(round(b.cy))), int(b.r * 1.6), 255, -1)
    for b in getattr(find_badges, "round_icons", []):          # 圆形图标：按圆切下来，图案留在里面
        r = int(np.ceil(b.r)) + 3
        x0, y0, x1, y1 = max(0, int(b.cx) - r), max(0, int(b.cy) - r), min(W, int(b.cx) + r + 1), min(H, int(b.cy) + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (np.hypot(xx - b.cx, yy - b.cy) <= b.r + 1.5).astype(np.uint8) * 255
        bw = int(2 * b.r + 3)
        icons.append(Icon(x0, y0, np.dstack([img_bgr[y0:y1, x0:x1], cv2.GaussianBlur(disc, (3, 3), 0)]), "图标",
                          (int(b.cx - b.r - 1), int(b.cy - b.r - 1), bw, bw), int(round(100 * min(1.0, b.rim))),
                          "圆形图标 · 整个圆连同里面的图案一起抠"))
        cv2.circle(erase, (int(round(b.cx)), int(round(b.cy))), int(b.r + 3), 255, -1)
        cv2.circle(avoid, (int(round(b.cx)), int(round(b.cy))), int(b.r * 1.15), 255, -1)
    return icons, erase, avoid


def attach_dashes(img_bgr, lines, full_mask):
    """文字识别常常漏掉行首 / 行尾的破折号"——"（它只是一条横线，不像字）。
    这里在每行文字的左右两边找：颜色和这行字一样、又细又长、高度在这行字中间的横线，
    找到就把"——"补进这行文字，并把横线也算进要抹掉的范围。"""
    H, W = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    for ln in lines:
        if ln.badge or ln.size_px < 10 or not ln.ink:
            continue
        s = ln.size_px
        col = np.array(ln.color[::-1], np.float32)                 # 文字色（BGR）
        ix0, iy0, ix1, iy1 = ln.ink
        ya, yb = int(iy0 + 0.25 * (iy1 - iy0)), int(iy0 + 0.8 * (iy1 - iy0))
        for side in ("left", "right"):
            xa, xb = (max(0, int(ix0 - 5.5 * s)), max(0, ix0 - 2)) if side == "left" else (min(W, ix1 + 2), min(W, int(ix1 + 5.5 * s)))
            if xb - xa < 0.8 * s or yb - ya < 3:
                continue
            band = f[ya:yb, xa:xb]
            like = (np.linalg.norm(band - col, axis=2) < 60).astype(np.uint8)
            like[full_mask[ya:yb, xa:xb] > 0] = 0                  # 已经属于别的文字的不算
            n, lab, stats, _ = cv2.connectedComponentsWithStats(like, connectivity=8)
            segs = [stats[i] for i in range(1, n)
                    if stats[i][cv2.CC_STAT_HEIGHT] <= max(3, 0.16 * s) and stats[i][cv2.CC_STAT_WIDTH] >= 0.45 * s]
            if not segs:
                continue
            # 破折号可能是一整条，也可能中间断开成两段；它们得在同一高度、并且紧挨着文字
            near_seg = max(segs, key=lambda sg: sg[0] + sg[2]) if side == "left" else min(segs, key=lambda sg: sg[0])
            yc = near_seg[1] + near_seg[3] / 2                  # 以最靠近文字的那一段为准
            segs = [sg for sg in segs if abs(sg[1] + sg[3] / 2 - yc) <= 2]
            if not segs:
                continue
            x_lo = min(sg[0] for sg in segs)
            x_hi = max(sg[0] + sg[2] for sg in segs)
            total = sum(sg[2] for sg in segs)
            gap = (xb - xa - x_hi) if side == "left" else x_lo     # 横线到文字的空隙
            far_end_cut = (x_lo <= 1) if side == "left" else (x_hi >= xb - xa - 1)   # 横线一直伸到搜索范围之外：太长，不是破折号
            if os.environ.get('DASHDBG'): print('   dash?', ln.text[:8], side, 'gap', gap, 'total', total, 's', round(s), 'cut', far_end_cut, [tuple(int(v) for v in sg[:4]) for sg in segs])
            if gap > 0.8 * s or far_end_cut or not 0.8 * s <= total <= 4.6 * s:
                continue
            # 上下必须是空的：是孤立的横线，而不是某个大色块、下划线的一段
            y_top, y_bot = min(sg[1] for sg in segs), max(sg[1] + sg[3] for sg in segs)
            above = like[max(0, y_top - 4):max(0, y_top - 1), x_lo:x_hi]
            below = like[y_bot + 1:y_bot + 4, x_lo:x_hi]
            if (above.size and above.mean() > 0.1) or (below.size and below.mean() > 0.1):
                continue
            dash = "—" * int(np.clip(round((x_hi - x_lo) / s), 1, 4))        # 原图的横线有几个字宽，就补几个破折号
            dash_w = float(x_hi - x_lo)
            m = np.zeros(like.shape, np.uint8)
            m[y_top:y_bot, x_lo:x_hi] = like[y_top:y_bot, x_lo:x_hi] * 255
            k = max(3, int(round(s * 0.08)) | 1)
            full_mask[ya:yb, xa:xb] = np.maximum(full_mask[ya:yb, xa:xb], cv2.dilate(m, np.ones((k, k), np.uint8)))
            if side == "left":
                ln.text = dash + ln.text
                ln.ink = (xa + x_lo, iy0, ix1, iy1)
                ix0 = ln.ink[0]
            else:
                ln.text = ln.text + dash
                ln.ink = (ix0, iy0, xa + x_hi, iy1)
                ix1 = xa + x_hi
            ln.chars = None
            if ln.runs:                                            # 破折号并进紧挨着的那一段
                r = list(ln.runs)
                i = 0 if side == "left" else -1
                r[i] = ((dash + r[i][0]) if side == "left" else (r[i][0] + dash),) + tuple(r[i][1:])
                ln.runs = r
            ln.fit = 0.0


def attach_trailing_punct(img_bgr, lines, full_mask):
    """文字识别有时漏掉行尾的逗号、句号（它们又小又贴着底边）。
    在每行文字右边紧挨着的位置找：颜色和这行字一样、很小、位于这行字下半部分的一个墨点，找到就补上。"""
    H, W = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    for ln in lines:
        if ln.badge or ln.size_px < 14 or not ln.ink or ln.text[-1:] in "，。、；：,.;:!?！？":
            continue
        if not any("\u4e00" <= c <= "\u9fff" for c in ln.text):
            continue                                           # 只处理中文行（英文标志旁边的装饰容易被误认）
        if ln.text.count("“") > ln.text.count("”"):            # 行尾漏掉的右引号：在这行字上半部分找
            s = ln.size_px
            ix0, iy0, ix1, iy1 = ln.ink
            xa, xb = min(W, ix1 + 1), min(W, int(ix1 + 0.75 * s))
            ya, yb = max(0, int(iy0 - 0.1 * s)), int(iy0 + 0.5 * (iy1 - iy0))
            if xb - xa >= 4 and yb - ya >= 4:
                col = np.array(ln.color[::-1], np.float32)
                like = (np.linalg.norm(f[ya:yb, xa:xb] - col, axis=2) < 70).astype(np.uint8)
                like[full_mask[ya:yb, xa:xb] > 0] = 0
                n, lab, stats, _ = cv2.connectedComponentsWithStats(like, connectivity=8)
                marks = [stats[i] for i in range(1, n)
                         if 0.08 * s <= max(stats[i][2], stats[i][3]) <= 0.45 * s and stats[i][4] >= 6
                         and stats[i][0] + stats[i][2] < xb - xa - 1 and stats[i][0] <= 0.5 * s]
                if 1 <= len(marks) <= 2:
                    m = np.zeros(like.shape, np.uint8)
                    for x, y, w, h, _ in marks:
                        m[y:y + h, x:x + w] = like[y:y + h, x:x + w] * 255
                    full_mask[ya:yb, xa:xb] = np.maximum(full_mask[ya:yb, xa:xb], cv2.dilate(m, np.ones((5, 5), np.uint8)))
                    ln.text += "”"
                    ln.ink = (ix0, iy0, xa + max(x + w for x, y, w, h, _ in marks), iy1)
                    ln.chars = None
                    if ln.runs:
                        r = list(ln.runs)
                        r[-1] = (r[-1][0] + "”",) + tuple(r[-1][1:])
                        ln.runs = r
            continue
        s = ln.size_px
        ix0, iy0, ix1, iy1 = ln.ink
        xa, xb = min(W, ix1 + 1), min(W, int(ix1 + 0.9 * s))
        ya, yb = max(0, int(iy0 + 0.35 * (iy1 - iy0))), min(H, int(iy1 + 0.3 * s))
        if xb - xa < 4 or yb - ya < 4:
            continue
        col = np.array(ln.color[::-1], np.float32)
        like = (np.linalg.norm(f[ya:yb, xa:xb] - col, axis=2) < 70).astype(np.uint8)
        like[full_mask[ya:yb, xa:xb] > 0] = 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(like, connectivity=8)
        best = None
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (0.08 * s <= max(w, h) <= 0.45 * s and area >= 6):
                continue
            if x + w >= xb - xa - 1 or y <= 0:                    # 贴着搜索范围的边：是别的东西的一部分
                continue
            if x > 0.55 * s:
                continue
            if best is None or x < best[0]:
                best = (x, y, w, h, area)
        if best is None:
            continue
        x, y, w, h, area = best
        hollow = area < 0.6 * w * h and abs(w - h) <= 0.25 * max(w, h)   # 中空的小圆圈是句号
        mark = "。" if hollow else "，"
        m = (lab == lab[y + h // 2, x + w // 2]).astype(np.uint8) * 255 if lab[y + h // 2, x + w // 2] else np.zeros(like.shape, np.uint8)
        m[y:y + h, x:x + w] = np.maximum(m[y:y + h, x:x + w], like[y:y + h, x:x + w] * 255)
        full_mask[ya:yb, xa:xb] = np.maximum(full_mask[ya:yb, xa:xb], cv2.dilate(m, np.ones((5, 5), np.uint8)))
        ln.text += mark
        ln.ink = (ix0, iy0, xa + x + w, iy1)
        ln.chars = None
        if ln.runs:
            r = list(ln.runs)
            r[-1] = (r[-1][0] + mark,) + tuple(r[-1][1:])
            ln.runs = r


def unify_sizes(lines, tol=0.085):
    """同一页上"差不多大"的字统一成同一个字号。
    每行的字号是单独量出来的，同一级别的文字会量出 12、12.5、13 这样的小差别，放在一起就显得不整齐。
    做法：把字号从小到大排，相邻差距在 8.5% 以内的归为一组，每组都改成这一组的中位数。"""
    todo = [ln for ln in lines if not ln.badge and ln.size_px > 0]
    if len(todo) < 2:
        return
    todo.sort(key=lambda l: l.size_px)
    groups, cur = [], [todo[0]]
    for ln in todo[1:]:
        if ln.size_px <= cur[0].size_px * (1 + 1.6 * tol) and ln.size_px <= cur[-1].size_px * (1 + tol):
            cur.append(ln)
        else:
            groups.append(cur)
            cur = [ln]
    groups.append(cur)
    for g in groups:
        target = float(np.median([l.size_px for l in g]))
        for ln in g:
            k = target / ln.size_px
            ln.size_px = target
            if ln.runs:
                ln.runs = [(t, c, b, s * k) for t, c, b, s in ln.runs]


def _clean_lines(lines, W):
    """OCR 结果的两类常见问题：
      ① 把"℃"认成"°℃"或"°°C"这种重复的符号
      ② 把一个图标认成一个孤零零的汉字（旁边同一行没有别的字）——这种不当文字处理，留给图标检测"""
    out = []
    for ln in lines:
        fixed = ln.text.replace("°℃", "℃").replace("°°", "°").replace("℃C", "℃")
        if fixed != ln.text:
            ln.text, ln.chars = fixed, None                    # 字数变了，逐字位置作废
        if len(ln.text) == 1 and "\u4e00" <= ln.text <= "\u9fff":
            cy = ln.y + ln.h / 2
            near = any(o is not ln and abs((o.y + o.h / 2) - cy) < 0.6 * ln.h
                       and min(abs(o.x - (ln.x + ln.w)), abs(ln.x - (o.x + o.w))) < 1.5 * ln.h for o in lines)
            if not near:
                continue
        out.append(ln)
    # 竖排的字（坐标轴旁边的"风险程度"）：拆成一个字一行，这样字号和位置才量得准
    res = []
    for ln in out:
        n = len(ln.text)
        if n >= 2 and ln.h >= 2.2 * ln.w and ln.h >= 0.8 * n * ln.w and all("\u4e00" <= c <= "\u9fff" for c in ln.text):
            step = ln.h / n
            for i, c in enumerate(ln.text):
                one = Line(c, ln.x, int(round(ln.y + i * step)), ln.w, int(round(step)))
                one.score = ln.score
                res.append(one)
        else:
            res.append(ln)
    return res


def _manual_element(img_bgr, box, kind, text_mask):
    """人工画的框 → 一个可移动的元素。返回 (Icon, 整页大小的布尔掩膜, 四周是否是干净底色)。
    框的四条边如果是同一种底色，就只取框里"和底色不一样"的部分（透明背景）；否则整框当成一张矩形图片。
    图标框里的文字不算进来（文字还是文字）；图片框里的文字跟着图片走。"""
    H, W = img_bgr.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, min(W - 2, x)), max(0, min(H - 2, y))
    w, h = max(2, min(W - x, w)), max(2, min(H - y, h))
    roi = img_bgr[y:y + h, x:x + w].astype(np.float32)
    edge = np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]])
    bg = np.median(edge, axis=0)
    clean_edge = float((np.linalg.norm(edge - bg, axis=1) < 22).mean()) > 0.9
    sub = np.ones((h, w), bool)
    if clean_edge:
        fg = (np.linalg.norm(roi - bg, axis=2) > 14).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, cs, -1, 1, -1)                  # 把里面的洞填上（图标里的白色部分也属于图标）
        if filled.mean() > 0.04:
            sub = filled > 0
    if kind == "图标":
        no_text = sub & (text_mask[y:y + h, x:x + w] == 0)
        if no_text.sum() >= 12:
            sub = no_text
    sel = np.zeros((H, W), bool)
    sel[y:y + h, x:x + w] = sub
    alpha = cv2.GaussianBlur(sub.astype(np.uint8) * 255, (3, 3), 0)
    alpha[sub] = 255
    rgba = np.dstack([img_bgr[y:y + h, x:x + w], alpha])
    ys, xs = np.where(sub)
    tight = (int(x + xs.min()), int(y + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return Icon(x, y, rgba, kind, tight, 100, "手动调整"), sel, clean_edge


def process_image(path, engine, lang, debug_dir=None, with_icons=True, inpaint="auto", with_pictures=True, on_stage=None,
                  ocr=None, edits=None):
    """on_stage(阶段名, 数据)：每完成一个阶段回调一次，网页界面用它显示真实进度。
    ocr   ：上一次的文字识别结果（process_image.last_ocr），传进来就不用再识别一遍
    edits ：人工调整后"想要的元素清单" [{"box": [x,y,w,h], "kind": "图标"/"图片", "src": 自动识别时的框 或 None}]。
            src 对得上的自动元素原样保留；清单里没有的自动元素不再抠（留在底图里）；src 为空的按 box 重新抠"""
    """on_stage(阶段名, 数据)：每完成一个阶段回调一次，网页界面用它显示真实进度。"""
    def tell(name, data=None):
        if on_stage:
            on_stage(name, data)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)   # 兼容中文路径
    if img is None:
        raise RuntimeError(f"读不了图片: {path}")
    H, W = img.shape[:2]
    tell("ocr")
    if ocr is not None:
        lines, used = copy.deepcopy(ocr), "沿用上次的文字识别"
    else:
        lines, used = run_ocr(img, engine, lang)
    process_image.last_ocr = copy.deepcopy(lines)
    lines = _clean_lines(lines, W)
    badges, lines = find_badges(img, lines)                    # 序号先分出来：圆圈归图形，里面的字归文字
    mask = np.zeros((H, W), np.uint8)
    kept = [ln for ln in lines if analyze_line(img, ln, mask)]
    tell("text", [[int(l.ink[0]), int(l.ink[1]), int(l.ink[2] - l.ink[0]), int(l.ink[3] - l.ink[1]), int(round(l.score * 100)), l.text] for l in kept])
    zero = np.zeros((H, W), np.uint8)
    # 先用"全部文字"的掩膜找图标和图片，再决定哪些字属于图片内部
    # 第一轮：先找大块的照片 / 插画；第二轮：在它们之外找图标（照片里的小图案属于照片，不单独抠）
    pics, pic_mask, pic_sel, smooth = extract_pictures(img, mask, []) if with_pictures else ([], zero, [], zero)
    if with_icons and pics:
        cand, _ = extract_icons(img, mask)                     # 不考虑图片时的图标候选
        drop = [i for i, p in enumerate(pics)
                if max(p.box[2], p.box[3]) <= 0.08 * W and any(_iou(p.box, c.box) > 0.5 for c in cand)]
        if drop:                                               # 这几张"图片"其实是图标：从图片里拿掉，交给图标那一轮
            keep = [i for i in range(len(pics)) if i not in drop]
            pics, pic_sel = [pics[i] for i in keep], [pic_sel[i] for i in keep]
            pic_mask = np.zeros((H, W), np.uint8)
            for sel in pic_sel:
                pic_mask[sel] = 255
            pic_mask = cv2.dilate(pic_mask, np.ones((5, 5), np.uint8))
            smooth = cv2.bitwise_and(smooth, pic_mask)
    tell("picture", [[int(v) for v in p.box] + [p.score, p.why] for p in pics])
    rejected = []
    if with_icons and getattr(find_badges, "round_icons", None):
        cand0, _ = extract_icons(img, mask, pic_mask > 0)
        def is_core(b):
            for c in cand0:
                x, y, w, h = c.box
                if x <= b.cx <= x + w and y <= b.cy <= y + h and min(w, h) >= 2 * b.r * 1.35 and max(w, h) <= 2 * b.r * 3:
                    return True
            return False
        find_badges.round_icons = [b for b in find_badges.round_icons if not is_core(b)]
    b_icons, b_erase, b_avoid = badge_icons(img, badges)
    if pics:                                                   # 已经按圆形图标处理的，不再当图片
        rb = [ic.box for ic in b_icons]
        keep = [i for i, p in enumerate(pics) if not any(_iou(p.box, r_) > 0.4 for r_ in rb)]
        if len(keep) < len(pics):
            pics, pic_sel = [pics[i] for i in keep], [pic_sel[i] for i in keep]
            pic_mask = np.zeros((H, W), np.uint8)
            for sel in pic_sel:
                pic_mask[sel] = 255
            pic_mask = cv2.dilate(pic_mask, np.ones((5, 5), np.uint8))
            smooth = cv2.bitwise_and(smooth, pic_mask)
    if with_pictures:                                          # 组合插画：几样东西叠成的一幅图，整簇抠，免得被拆成碎片
        g_all = getattr(_flat_graphics, "bars", None)
        taken0 = np.maximum(pic_mask, b_erase)
        if g_all is not None and g_all.shape == taken0.shape:
            taken0 = np.maximum(taken0, g_all)
        for sel_l, _c in getattr(_flat_graphics, "labels", []):
            if sel_l.shape == taken0.shape:
                taken0[sel_l] = 255                            # 压着文字的色块（标签）不算插画的一部分
        for ic, sel in find_illustrations(img, mask, taken0):
            pics.append(ic)
            pic_sel.append(sel)
            pic_mask[cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255
            smooth[sel] = 255
        tell("picture", [[int(v) for v in p.box] + [p.score, p.why] for p in pics])
    if with_pictures:                                          # 第三轮：补找前面漏掉的配图（浅色设备图、小图表……）
        cand_i, cand_m = extract_icons(img, mask, pic_mask > 0, None, b_avoid) if with_icons else ([], zero)
        for ic, sel in find_leftover_pictures(img, mask, np.maximum(np.maximum(pic_mask, b_erase), cand_m)):
            pics.append(ic)
            pic_sel.append(sel)
            pic_mask[cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255
            smooth[sel] = 255                                  # 四周是干净底色，抠走后直接平滑填充
    photo_zone = np.zeros((H, W), bool)                        # 贴着页面边缘的照片区域：里面不找淡彩图标
    for px, py, pw, ph in find_picture_regions(img, mask):
        photo_zone[py:py + ph, px:px + pw] = True
    bars = getattr(_flat_graphics, "bars", None)               # 时间轴和轴上的圆点：留在底图里，不当图标抠
    icon_excl = (pic_mask > 0) | (bars > 0) if bars is not None and bars.shape == pic_mask.shape else pic_mask > 0
    icons, icon_mask = extract_icons(img, mask, icon_excl, rejected, b_avoid, photo_zone) if with_icons else ([], zero)
    icons = b_icons + icons
    icon_mask = np.maximum(icon_mask, b_erase)
    if edits is not None:
        keep_src = {tuple(int(v) for v in e["src"]) for e in edits if e.get("src")}
        # 图片：清单里没有的不再抠
        keep_i = [i for i, p in enumerate(pics) if tuple(int(v) for v in p.box) in keep_src]
        pics, pic_sel = [pics[i] for i in keep_i], [pic_sel[i] for i in keep_i]
        pic_mask = np.zeros((H, W), np.uint8)
        for sel in pic_sel:
            pic_mask[cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255
        smooth = cv2.bitwise_and(smooth, pic_mask)
        # 图标：去掉的那些，把它们占的位置从"要抹掉的范围"里拿出来；留下的图标如果和它挨着，再补回去
        gone = [ic for ic in icons if tuple(int(v) for v in ic.box) not in keep_src]
        icons = [ic for ic in icons if tuple(int(v) for v in ic.box) in keep_src]
        for ic in gone:
            bx, by, bw, bh = [int(v) for v in ic.box]
            icon_mask[max(0, by - 5):by + bh + 5, max(0, bx - 5):bx + bw + 5] = 0
        if gone:
            for ic in icons:
                hh, ww = ic.rgba.shape[:2]
                a = cv2.dilate((ic.rgba[..., 3] > 8).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
                icon_mask[ic.y:ic.y + hh, ic.x:ic.x + ww] = np.maximum(icon_mask[ic.y:ic.y + hh, ic.x:ic.x + ww], a)
        # 人工画的 / 改过大小的框
        for e in edits:
            if e.get("src"):
                continue
            el, sel, clean_edge = _manual_element(img, e["box"], e.get("kind", "图标"), mask)
            if el.kind == "图片":
                pics.append(el)
                pic_sel.append(sel)
                pic_mask[cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255
                if clean_edge:
                    smooth[sel] = 255
            else:
                icons.append(el)
                icon_mask[cv2.dilate(sel.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 255
    tell("icon", [[int(v) for v in ic.box] + [ic.score, ic.why] for ic in icons])
    tell("rejected", [[int(v) for v in ic.box] + [ic.score, ic.why] for ic in rejected])
    pictures = find_picture_regions(img, mask)                 # 贴着画面边缘、没法整张抠出来的照片区域
    free = cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0
    def inside(ln):
        if ln.badge:
            return False
        cx, cy = int((ln.ink[0] + ln.ink[2]) / 2), int((ln.ink[1] + ln.ink[3]) / 2)
        if any(sel[min(H - 1, cy), min(W - 1, cx)] for sel in pic_sel):
            return True                                        # 在抠出来的图片里：字跟着图片走
        if any(px <= cx <= px + pw and py <= cy <= py + ph for px, py, pw, ph in pictures):
            return True
        return False
    skipped = [ln for ln in kept if inside(ln)]
    if skipped:
        mask = np.zeros((H, W), np.uint8)
        kept = [ln for ln in kept if not inside(ln) and analyze_line(img, ln, mask)]
    attach_dashes(img, kept, mask)                            # 放在最后：前面可能重建过文字掩膜
    attach_trailing_punct(img, kept, mask)
    icon_mask = np.maximum(icon_mask, pic_mask)
    icons = pics + icons
    tell("erase")
    clean, how = erase_text(img, np.maximum(mask, icon_mask), inpaint, smooth, pic_mask)
    if bars is not None and bars.shape == mask.shape:          # 时间轴留在底图里：抹掉旁边的图片时不能蹭掉它的边
        keep_bar = (bars > 0) & (cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0)
        clean[keep_bar] = img[keep_bar]
    for sel, col in getattr(_flat_graphics, "labels", []):     # 纯色标签上的字：直接用标签的颜色盖掉
        if sel.shape != mask.shape or (icon_mask[sel] > 0).mean() > 0.3:
            continue
        hole = sel & (cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0)
        if not hole.any():
            continue
        # 字几乎占满整个标签时，普通补全会把标签外面的底色"抹"进来。这里把标签外面先换成标签的颜色，
        # 再补字留下的洞：补出来的只会是标签自己的颜色（带渐变的也能接上）
        lx, ly, lw, lh = cv2.boundingRect(sel.astype(np.uint8))
        crop = img[ly:ly + lh, lx:lx + lw].copy()
        s_in, h_in = sel[ly:ly + lh, lx:lx + lw], hole[ly:ly + lh, lx:lx + lw]
        crop[~s_in] = np.clip(col, 0, 255).astype(np.uint8)
        fixed = cv2.inpaint(crop, h_in.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        clean[ly:ly + lh, lx:lx + lw][h_in] = fixed[h_in]
    unify_sizes(kept)
    refit_widths(kept)
    detect_italic(kept)
    blocks = group_lines(kept)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        dbg = img.copy()
        for ln in kept:
            cv2.rectangle(dbg, ln.ink[:2], ln.ink[2:], (0, 0, 255), 2)
        for px, py, pw, ph in pictures:
            cv2.rectangle(dbg, (px, py), (px + pw, py + ph), (255, 0, 255), 2)
        for ic in icons:
            cv2.rectangle(dbg, (ic.x, ic.y), (ic.x + ic.rgba.shape[1], ic.y + ic.rgba.shape[0]),
                          (0, 180, 0) if ic.kind == "图标" else (255, 120, 0), 2)
        for name, im in (("1_ocr", dbg), ("2_mask", np.maximum(mask, icon_mask)), ("3_clean", clean)):
            cv2.imencode(".png", im)[1].tofile(os.path.join(debug_dir, f"{stem}_{name}.png"))
    print(f"  {os.path.basename(path)}: {used} 识别 {len(lines)} 行，转换 {len(kept)} 行（图片内的字跳过 {len(skipped)} 行），合并为 {len(blocks)} 个文本框，其中序号 {len(badges)} 个；抠出 {len(icons) - len(pics)} 个图标、{len(pics)} 张图片（另有 {len(rejected)} 个候选分数不够未抠），抹除方式 {how}")
    process_image.last_rejected = rejected                     # 分数不够、没有抠的候选（网页界面会用虚线标出来）
    return clean, blocks, icons, W, H


def collect_inputs(inputs):
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(exts))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="幻灯片图片 → 文字可编辑的 PPTX")
    ap.add_argument("inputs", nargs="+", help="图片文件或文件夹（每张图一页）")
    ap.add_argument("-o", "--output", default="output.pptx")
    ap.add_argument("--engine", choices=["auto", "rapid", "tesseract"], default="auto")
    ap.add_argument("--lang", default="chi_sim+eng", help="tesseract 语言（仅备用引擎用）")
    ap.add_argument("--font", default=default_ppt_font(), help="PPT 里文本框使用的字体")
    ap.add_argument("--inpaint", choices=["auto", "lama", "telea"], default="auto", help="抹字补背景的方式")
    ap.add_argument("--no-pictures", action="store_true", help="不抠照片 / 插画，留在底图里")
    ap.add_argument("--no-icons", action="store_true", help="不抠图标，图标留在底图里")
    ap.add_argument("--debug", action="store_true", help="输出 OCR 框、掩膜、抹字后的底图")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("没有找到图片")
    prs = None
    for path in files:
        clean, blocks, icons, W, H = process_image(
            path, args.engine, args.lang,
            os.path.join(os.path.dirname(os.path.abspath(args.output)), "debug") if args.debug else None,
            with_icons=not args.no_icons, inpaint=args.inpaint, with_pictures=not args.no_pictures)
        if prs is None:                                       # 用第一张图的比例决定页面尺寸
            prs = Presentation()
            prs.slide_width = SLIDE_W_EMU
            prs.slide_height = int(SLIDE_W_EMU * H / W)
        add_slide(prs, clean, blocks, icons, W, H, args.font)
    prs.save(args.output)
    print(f"已生成: {args.output}（{len(files)} 页）")


if __name__ == "__main__":
    main()
