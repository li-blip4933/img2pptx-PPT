"""Conservative detection of standalone arrow graphics on slide images.

The detector is deliberately independent of the OCR/PowerPoint pipeline.  It
uses connected components and a small set of arrow-shape checks, then returns
local masks that callers can crop as image assets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass
class ArrowRegion:
    """A detected arrow; ``mask`` is a bool array aligned to ``box``."""

    box: tuple[int, int, int, int]
    mask: np.ndarray = field(repr=False, compare=False)
    direction: str = "right"
    color: str = "gray"
    score: float = 0.0
    text_changes: tuple["ArrowTextChange", ...] = ()


@dataclass(frozen=True)
class ArrowTextChange:
    """OCR edit recommended because a detected arrow occupied an edge glyph."""

    line_index: int
    removed_text: str
    remaining_text: str


_ARROW_CHARS = frozenset("↑↓←→↖↗↘↙⇧⇩⇦⇨➔➜➝➞➤")
_SIZE = 64
_POLYGONS = {
    "right": ((0, .35), (.62, .35), (.62, .08), (1, .50),
              (.62, .92), (.62, .65), (0, .65)),
    "left": ((1, .35), (.38, .35), (.38, .08), (0, .50),
             (.38, .92), (.38, .65), (1, .65)),
    "up": ((.35, 1), (.35, .38), (.08, .38), (.50, 0),
           (.92, .38), (.65, .38), (.65, 1)),
    "down": ((.35, 0), (.35, .62), (.08, .62), (.50, 1),
             (.92, .62), (.65, .62), (.65, 0)),
}


def _templates():
    result = {}
    for direction, points in _POLYGONS.items():
        poly = np.asarray([[(round(x * (_SIZE - 1)), round(y * (_SIZE - 1)))
                            for x, y in points]], dtype=np.int32)
        mask = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
        cv2.fillPoly(mask, poly, 1)
        result[direction] = mask.astype(bool)
    return result


_TEMPLATES = _templates()


def _box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / max(1, union)


def _shape_score(mask, bbox):
    x, y, w, h = bbox
    aspect = w / max(1, h)
    if w >= h:
        if aspect < 1.18:
            return None
        allowed = {"right", "left"}
    else:
        if aspect > 0.85:
            return None
        allowed = {"up", "down"}

    # Fill the outer contour for classification so thin, anti-aliased outlines
    # and filled arrows use the same geometry check. Keep the original pixels
    # in ArrowRegion.mask for image extraction.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    if not contours:
        return None
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    normalized = cv2.resize(filled, (_SIZE, _SIZE),
                            interpolation=cv2.INTER_NEAREST) > 0
    scores = {}
    for direction in allowed:
        template = _TEMPLATES[direction]
        scores[direction] = float(np.count_nonzero(normalized & template) /
                                  max(1, np.count_nonzero(normalized | template)))
    direction = max(scores, key=scores.get)
    return direction, scores[direction]


def _long_connector(mask, bbox, image_shape):
    """Recognize a long, thin return-flow arrow with a single arrow tip."""
    x, y, w, h = bbox
    image_h, image_w = image_shape[:2]
    if w < .28 * image_w or w / max(1, h) < 7 or h < max(10, .018 * image_h):
        return False
    if h > .12 * image_h:
        return False

    # A return connector has a long horizontal rail and a small tapered end.
    row_counts = np.count_nonzero(mask, axis=1)
    if len(row_counts) < 8:
        return False
    # The rail may sit close to either edge of the component (the source
    # return arrow has its horizontal run near the bottom of the crop).
    if np.max(row_counts) < .55 * w:
        return False
    extreme = row_counts[:max(2, int(.12 * h))]
    if np.max(extreme) > .22 * w:
        return False
    return True


def _character_spans(line):
    text = str(getattr(line, "text", ""))
    x = int(getattr(line, "x", 0))
    width = max(1, int(getattr(line, "w", 1)))
    chars = getattr(line, "chars", None)
    if chars and len(chars) == len(text):
        return [(int(span[0]), int(span[1])) for span in chars]
    weights = [1.0 if ord(char) > 255 else
               (0.28 if char.isspace() else 0.55 if char.isalnum() else 0.48)
               for char in text]
    total = sum(weights) or 1.0
    spans, at = [], float(x)
    for weight in weights:
        end = at + width * weight / total
        spans.append((round(at), round(end)))
        at = end
    return spans


def _line_edge_glyph(line, bbox):
    """Return the edge character covered by an arrow, or ``None``.

    Interior arrow punctuation such as ``75%→80%`` is kept as text. A
    non-arrow glyph is eligible only at a line edge and only when its OCR box
    strongly overlaps the geometric arrow component (covers common ``↑`` →
    ``一`` OCR confusions without dropping ordinary label characters).
    """
    text = str(getattr(line, "text", ""))
    if not text:
        return None
    spans = _character_spans(line)
    x, y, w, h = bbox
    lx, ly = int(getattr(line, "x", 0)), int(getattr(line, "y", 0))
    lh = max(1, int(getattr(line, "h", 1)))
    y_overlap = max(0, min(y + h, ly + lh) - max(y, ly)) / max(1, min(h, lh))
    if y_overlap < .30:
        return None

    possible = []
    for index, (start, end) in enumerate(spans):
        overlap = max(0, min(x + w, end) - max(x, start))
        ratio = overlap / max(1, min(w, end - start))
        if ratio >= .35:
            possible.append((ratio, index))
    if not possible:
        return None
    _, index = max(possible)
    if index not in (0, len(text) - 1):
        return None
    char = text[index]
    if char not in _ARROW_CHARS:
        # A near-horizontal dash is commonly used for the upward arrow OCR
        # confusion. Require a much tighter overlap before removing it.
        dash_confusion = char in "一—–-" and max(possible)[0] >= .72
        up_confusion = char == "个" and h >= 1.4 * w and max(possible)[0] >= .85
        if not (dash_confusion or up_confusion):
            return None
    return index


def _candidate_text_change(lines, bbox):
    matched = []
    x, y, w, h = bbox
    for line_index, line in enumerate(lines):
        lx, ly = int(getattr(line, "x", 0)), int(getattr(line, "y", 0))
        lw, lh = max(1, int(getattr(line, "w", 1))), max(1, int(getattr(line, "h", 1)))
        if _box_iou(bbox, (lx, ly, lw, lh)) <= 0:
            continue
        char_index = _line_edge_glyph(line, bbox)
        if char_index is None:
            # A text box may surround a nearby icon. It is safe to keep the
            # icon, but do not delete any OCR text unless its edge glyph maps
            # to the arrow pixels.
            continue
        matched.append((line_index, char_index))
    return matched


def _overlaps_ordinary_ocr_glyph(lines, bbox):
    """Reject a shape component that is actually part of a text glyph."""
    x, y, w, h = bbox
    for line in lines or ():
        text = str(getattr(line, "text", ""))
        if not text:
            continue
        lx, ly = int(getattr(line, "x", 0)), int(getattr(line, "y", 0))
        lw, lh = max(1, int(getattr(line, "w", 1))), max(1, int(getattr(line, "h", 1)))
        if _box_iou(bbox, (lx, ly, lw, lh)) <= 0:
            continue
        y_overlap = max(0, min(y + h, ly + lh) - max(y, ly)) / max(1, min(h, lh))
        if y_overlap < .3:
            continue
        spans = _character_spans(line)
        overlaps = []
        for index, (start, end) in enumerate(spans):
            amount = max(0, min(x + w, end) - max(x, start))
            ratio = amount / max(1, min(w, end - start))
            if ratio >= .35:
                overlaps.append(index)
        if overlaps and _line_edge_glyph(line, bbox) is None:
            return True
    return False


def _apply_text_changes(lines, regions):
    edits = {}
    for region in regions:
        for line_index, char_index in _candidate_text_change(lines, region.box):
            edits.setdefault(line_index, set()).add(char_index)

    filtered = []
    changes_by_region = {id(region): [] for region in regions}
    for line_index, line in enumerate(lines):
        removed = edits.get(line_index, set())
        if not removed:
            filtered.append(line)
            continue
        text = str(getattr(line, "text", ""))
        # Only edge glyphs are eligible, so repeatedly trim from either side.
        left, right = 0, len(text)
        while left in removed:
            left += 1
        while right - 1 in removed:
            right -= 1
        if left >= right:
            continue
        if left == 0 and right == len(text):
            filtered.append(line)
            continue
        new_line = copy.copy(line)
        new_line.text = text[left:right]
        spans = _character_spans(line)
        old_x = int(getattr(line, "x", 0))
        old_right = old_x + int(getattr(line, "w", 1))
        if left:
            new_line.x = spans[left - 1][1]
        if right < len(text):
            old_right = spans[right][0]
        new_line.w = max(1, old_right - int(new_line.x))
        old_chars = getattr(line, "chars", None)
        if old_chars and len(old_chars) == len(text):
            new_line.chars = list(old_chars[left:right])
        filtered.append(new_line)
        change = ArrowTextChange(line_index, text[:left] + text[right:], new_line.text)
        for region in regions:
            if _line_edge_glyph(line, region.box) in removed:
                changes_by_region[id(region)].append(change)
    for region in regions:
        region.text_changes = tuple(changes_by_region[id(region)])
    return filtered


def detect_graphic_arrows(img_bgr: np.ndarray, ocr_lines: Sequence = ()):
    """Return ``(filtered_lines, arrow_regions)`` for high-confidence graphics.

    ``ocr_lines`` is duck-typed (``text/x/y/w/h`` and optional ``chars``).
    The returned lines are shallow copies only where a verified arrow glyph
    was trimmed; all other line objects are preserved. Each region's local
    bool mask and box use source-image pixel coordinates.
    """
    if img_bgr is None or img_bgr.ndim < 2:
        raise ValueError("img_bgr must be a non-empty color image")
    height, width = img_bgr.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("img_bgr must be a non-empty color image")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = (
        ("orange", ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 28) &
                     (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 100))),
        ("gray", ((hsv[:, :, 1] < 85) & (hsv[:, :, 2] >= 140) &
                  (hsv[:, :, 2] <= 240))),
        ("blue", ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 135) &
                   (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 70))),
    )

    regions = []
    min_area = max(25, round(width * height * .000025))
    min_dim = max(5, round(min(height, width) * .006))
    for color, binary in masks:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            x, y, w, h, area = map(int, stats[label])
            if area < min_area or min(w, h) < min_dim:
                continue
            local = labels[y:y + h, x:x + w] == label
            bbox = (x, y, w, h)
            if _long_connector(local, bbox, img_bgr.shape):
                direction, score = "right", .85
            else:
                shape = _shape_score(local, bbox)
                if shape is None:
                    continue
                direction, score = shape
                threshold = .50 if color == "gray" else .53
                if direction in ("up", "down"):
                    threshold = .64
                if score < threshold:
                    continue
            # Do not convert an inline OCR arrow into an icon. OCR at an edge
            # may be trimmed below; an interior arrow remains editable text.
            interior_punctuation = False
            for line in ocr_lines or ():
                text = str(getattr(line, "text", ""))
                if not text:
                    continue
                ly = int(getattr(line, "y", 0))
                lh = max(1, int(getattr(line, "h", 1)))
                y_overlap = max(0, min(y + h, ly + lh) - max(y, ly)) / max(
                    1, min(h, lh))
                if y_overlap < .30:
                    continue
                spans = _character_spans(line)
                for index, char in enumerate(text):
                    if char not in _ARROW_CHARS or index in (0, len(text) - 1):
                        continue
                    start, end = spans[index]
                    if max(0, min(x + w, end) - max(x, start)) >= .35 * min(w, end - start):
                        interior_punctuation = True
                        break
                if interior_punctuation:
                    break
            if interior_punctuation:
                continue
            if _overlaps_ordinary_ocr_glyph(ocr_lines, bbox):
                continue
            candidate = ArrowRegion(bbox, local.copy(), direction, color, score)
            duplicate = next((prior for prior in regions
                              if _box_iou(prior.box, bbox) >= .55), None)
            if duplicate is None:
                regions.append(candidate)
            elif candidate.score > duplicate.score:
                regions[regions.index(duplicate)] = candidate

    regions.sort(key=lambda arrow: (arrow.box[1], arrow.box[0]))
    filtered_lines = _apply_text_changes(list(ocr_lines or ()), regions)
    return filtered_lines, regions
