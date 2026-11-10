#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning.py —— 从人工修改里学习，让识别越用越准

每次在网页上"应用修改"，app.py 调 record_page()，把这一页的修改记成一条"教训"：
  · 图标 / 图片：自动识别出来的每个候选框，你留下了、删掉了、改了类别，还是自己补画了一个
    —— 每个框算一组外观特征（大小、形状、颜色、和周围的反差……），连同你的决定一起存下来
  · 文字：你把哪个字改成了哪个字（"目我"→"自我"）
  · 整页：同一张图下次再转，直接沿用你上次的修改
下一次识别时，process_image() 调 decide_elements()：用这些教训训练出来的小模型（逻辑回归 + 近邻投票），
和原来的规则分一起决定每个候选框留不留、是图标还是图片；text_fixes() 自动改掉你改过的错字。

所有数据存在程序目录下的 learning/ 文件夹（只在你自己电脑上）。删掉这个文件夹就回到"出厂"状态。
在线公开版默认不学习（不保存访客的图片和修改），除非设了 IMG2PPTX_LEARN=1。
"""
import difflib
import hashlib
import json
import os
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEARN_DIR = os.environ.get("IMG2PPTX_LEARN_DIR") or os.path.join(HERE, "learning")
_LOCK = threading.Lock()
_CACHE = {}

MIN_SAMPLES = 12          # 至少攒够这么多条图形教训（两类都有）才开始用学到的模型
MODEL_WEIGHT = 0.5        # 学到的模型和原规则各占多少


def enabled():
    if os.environ.get("IMG2PPTX_LEARN") == "0":
        return False
    if os.environ.get("IMG2PPTX_PUBLIC") == "1" and os.environ.get("IMG2PPTX_LEARN") != "1":
        return False
    return True


def _path(name):
    return os.path.join(LEARN_DIR, name)


def image_key(img_bgr):
    """一张图的指纹：同一张图再转时认出来。缩小后取哈希，对重新保存 / 轻微压缩不敏感。"""
    small = cv2.resize(img_bgr, (64, 40), interpolation=cv2.INTER_AREA)
    small = (small // 16).astype(np.uint8)
    return hashlib.sha1(small.tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# 特征：任何一个框（自动候选或人工画的）都用同一套量法
# ----------------------------------------------------------------------------
FEATURE_NAMES = ["宽占比", "高占比", "面积占比", "长宽比", "饱和度均值", "饱和度离散", "亮度均值", "亮度离散",
                 "边缘密度", "颜色数", "前景占比", "和外圈反差", "外圈均匀度", "文字覆盖", "贴边", "自动分"]


def box_features(img_bgr, box, text_mask=None, auto_score=None):
    H, W = img_bgr.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, x), max(0, y)
    w, h = max(2, min(w, W - x)), max(2, min(h, H - y))
    crop = img_bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pad = max(3, int(0.15 * max(w, h)))
    X0, Y0, X1, Y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    outer = img_bgr[Y0:Y1, X0:X1].astype(np.float32)
    ring = np.ones(outer.shape[:2], bool)
    ring[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
    ring_px = outer[ring] if ring.any() else outer.reshape(-1, 3)
    ring_c = np.median(ring_px, axis=0)
    ring_uni = float(np.mean(np.linalg.norm(ring_px - ring_c, axis=1))) / 100.0
    d = np.linalg.norm(crop.astype(np.float32) - ring_c, axis=2)
    fg = float((d > 30).mean())
    contrast = float(np.percentile(d, 90)) / 255.0
    edges = cv2.Canny(gray, 60, 160)
    q = (crop // 32).reshape(-1, 3).astype(np.int32)
    codes = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    counts = np.bincount(codes, minlength=512)
    n_colors = float((counts > 0.01 * len(codes)).sum()) / 20.0
    tcover = 0.0
    if text_mask is not None and text_mask.shape[:2] == (H, W):
        tcover = float((text_mask[y:y + h, x:x + w] > 0).mean())
    edge_touch = float(x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2)
    s = (auto_score if auto_score is not None else 50) / 100.0
    return [w / W, h / H, (w * h) / float(W * H), float(np.log(w / float(h))),
            float(hsv[..., 1].mean()) / 255.0, float(hsv[..., 1].std()) / 255.0,
            float(hsv[..., 2].mean()) / 255.0, float(gray.std()) / 128.0,
            float((edges > 0).mean()), n_colors, fg, contrast, ring_uni, tcover, edge_touch, s]


# ----------------------------------------------------------------------------
# 存取
# ----------------------------------------------------------------------------
def _load_json(name, default):
    try:
        with open(_path(name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(name, data):
    os.makedirs(LEARN_DIR, exist_ok=True)
    tmp = _path(name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _path(name))


def _samples():
    out = []
    try:
        with open(_path("samples.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / max(1e-6, aw * ah + bw * bh - inter)


# ----------------------------------------------------------------------------
# 记一条教训
# ----------------------------------------------------------------------------
def record_page(img_bgr, page, elements=None, text_edits=None, text_mask=None):
    """page：修改前这一页的识别结果（app.py 里 pages[i]：icon / picture / rejected / lines）。
    elements：修改后想要的图形清单 [{"box", "kind": 图标/图片, "src"}]，没改图形时为 None。
    text_edits：这次的改字记录。返回这次记下了几条。"""
    if not enabled():
        return 0
    key = image_key(img_bgr)
    now = time.time()
    rows = []
    if elements is not None:
        auto = [(r[:4], "图标", r[4] if len(r) > 4 else 60) for r in page.get("icon", [])] + \
               [(r[:4], "图片", r[4] if len(r) > 4 else 60) for r in page.get("picture", [])]
        rej = [(r[:4], r[4] if len(r) > 4 else 40) for r in page.get("rejected", [])]
        used = set()
        kept_src = {tuple(int(v) for v in e["src"]): e for e in elements if e.get("src")}
        new_boxes = [e for e in elements if not e.get("src")]
        changed = 0
        for box, kind, score in auto:
            e = kept_src.get(tuple(int(v) for v in box))
            if e is not None:
                rows.append({"f": box_features(img_bgr, box, text_mask, score), "keep": 1, "kind": kind,
                             "auto_kind": kind, "w": 1.0})
                continue
            # 没原样留下：被改了大小 / 类别（附近有新框），或者被删了
            match = max(new_boxes, key=lambda n: _iou(n["box"], box), default=None)
            if match is not None and _iou(match["box"], box) > 0.3:
                used.add(id(match))
                rows.append({"f": box_features(img_bgr, box, text_mask, score), "keep": 1, "kind": match["kind"],
                             "auto_kind": kind, "w": 1.0})
            else:
                rows.append({"f": box_features(img_bgr, box, text_mask, score), "keep": 0, "kind": kind, "w": 1.0})
            changed += 1
        for box, score in rej:
            match = max(new_boxes, key=lambda n: _iou(n["box"], box), default=None)
            if match is not None and _iou(match["box"], box) > 0.4:
                used.add(id(match))
                rows.append({"f": box_features(img_bgr, box, text_mask, score), "keep": 1, "kind": match["kind"], "w": 1.0})
                changed += 1
            else:                                              # 没被点出来：你也同意不抠（弱一点的证据）
                rows.append({"f": box_features(img_bgr, box, text_mask, score), "keep": 0, "kind": "图标", "w": 0.4})
        for e in new_boxes:                                    # 自己补画的：程序漏掉的
            if id(e) in used:
                continue
            rows.append({"f": box_features(img_bgr, e["box"], text_mask, 30), "keep": 1, "kind": e["kind"], "w": 1.0})
            changed += 1
        total = len(auto) + sum(1 for e in new_boxes if id(e) not in used)
        _append_history({"t": now, "key": key, "kind": "图形", "total": total, "changed": changed})
    text_pairs = _learn_text(page, text_edits or [])
    if text_edits:
        n_lines = len(page.get("lines", []))
        n_changed = sum(1 for e in text_edits if e.get("src"))
        _append_history({"t": now, "key": key, "kind": "文字", "total": n_lines, "changed": n_changed})
    with _LOCK:
        if rows:
            os.makedirs(LEARN_DIR, exist_ok=True)
            with open(_path("samples.jsonl"), "a", encoding="utf-8") as f:
                for r in rows:
                    r["t"], r["key"] = now, key
                    r.setdefault("src", "edit")
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 同一张图：记住最终想要的样子，下次直接用
        pages = _load_json("pages.json", {})
        rec = pages.get(key, {})
        if elements is not None:
            rec["elements"] = elements
        if text_edits:
            rec["texts"] = rec.get("texts", []) + list(text_edits)
        rec["t"] = now
        pages[key] = rec
        _save_json("pages.json", pages)
        _CACHE.clear()
    if rows:
        train()
    return len(rows) + text_pairs


def _append_history(item):
    with _LOCK:
        os.makedirs(LEARN_DIR, exist_ok=True)
        with open(_path("history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _learn_text(page, text_edits):
    """改字记录 → 错字对照表（只收短的替换：一到四个字，整行重写不算）。"""
    lines = {tuple(int(v) for v in r[:4]): r[4] for r in page.get("lines", [])}
    pairs = 0
    table = _load_json("text_fixes.json", {})
    for e in text_edits:
        if not e.get("src") or not e.get("text"):
            continue
        src = tuple(int(v) for v in e["src"])
        old = lines.get(src)
        if old is None:
            best = max(lines.items(), key=lambda kv: _iou(kv[0], src), default=None)
            if best is None or _iou(best[0], src) < 0.5:
                continue
            old = best[1]
        new = e["text"].strip()
        if old == new:
            continue
        sm = difflib.SequenceMatcher(None, old, new)
        ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
        if not ops or len(ops) > 3:
            continue                                           # 改动太多：是改写内容，不是认错字
        for tag, i1, i2, j1, j2 in ops:
            a, b = old[i1:i2], new[j1:j2]
            if tag != "replace" or not 1 <= len(a) <= 4 or len(b) > 6:
                continue
            # 带上前后各一个字，"目我"→"自我"比"目"→"自"安全得多
            lo, hi = max(0, i1 - 1), min(len(old), i2 + 1)
            wrong = old[lo:hi]
            right = old[lo:i1] + b + old[i2:hi]
            if len(wrong) < 2:
                continue
            k = wrong + "\t" + right
            table[k] = table.get(k, 0) + 1
            pairs += 1
    if pairs:
        with _LOCK:
            _save_json("text_fixes.json", table)
            _CACHE.clear()
    return pairs


# ----------------------------------------------------------------------------
# 训练：逻辑回归（留不留）+ 逻辑回归（图标还是图片）+ 近邻投票
# ----------------------------------------------------------------------------
def _fit_logreg(X, y, w, l2=0.05, iters=600, lr=0.5):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    beta = np.zeros(Z.shape[1])
    pos = (w * y).sum()
    neg = (w * (1 - y)).sum()
    cw = np.where(y > 0.5, 0.5 / max(pos, 1e-6), 0.5 / max(neg, 1e-6)) * w * len(y)   # 两类平衡
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ ((p - y) * cw) / len(y) + l2 * np.r_[beta[:-1], 0]
        beta -= lr * g
    return {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist()}


def _predict(m, X):
    Z = (np.asarray(X, float) - np.asarray(m["mu"])) / np.asarray(m["sd"])
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    return 1 / (1 + np.exp(-Z @ np.asarray(m["beta"])))


def _decide_acc(model_keep, X, y, w):
    """在一批有答案的样本上，算"规则分 + 模型"的综合判断有多准（和 decide_elements 用的是同一个算法）。"""
    rule = X[:, FEATURE_NAMES.index("自动分")]
    if model_keep is None:
        pred = rule >= 0.5
    else:
        pred = _predict(model_keep, X) >= 0.5                 # 规则分本身就是模型的一个输入，由模型直接拍板
    return float((w * (pred == (y > 0.5))).sum() / max(w.sum(), 1e-6))


def train():
    """用全部教训训练一个新版本，但先"考试"再上岗（自我进化，但不会越学越差）：
      · 考卷 = 你最近亲手改过的那部分样本（按时间留出最近的 30%，最多 200 条）
      · 新版本只用考卷以外的数据训练，在考卷上的成绩不低于现在这一版，才替换；否则继续用旧版
      · 同时记下"只用规则"的成绩，网页上能看到学习到底有没有帮上忙
    "下载时没改"带来的隐式样本只参与训练、不进考卷（它们只是"你没反对"，不如亲手改的可靠）。"""
    S = _samples()
    if not S:
        return None
    X = np.array([s["f"] for s in S], float)
    y = np.array([s["keep"] for s in S], float)
    w = np.array([s.get("w", 1.0) for s in S], float)
    t = np.array([s.get("t", 0) for s in S], float)
    explicit = np.array([s.get("src", "edit") == "edit" for s in S])
    old = _load_json("model.json", {})
    model = {"n": len(S), "n_explicit": int(explicit.sum()), "n_keep": int(y.sum()), "n_drop": int((1 - y).sum()),
             "t": time.time(), "version": old.get("version", 0)}
    ok_data = y.sum() >= 3 and (1 - y).sum() >= 3
    cand = _fit_logreg(X, y, w) if ok_data else None
    # 考试
    ex_idx = np.where(explicit)[0]
    exam = ex_idx[np.argsort(-t[ex_idx])][:min(200, max(6, int(0.3 * len(ex_idx))))] if len(ex_idx) >= 10 else np.array([], int)
    adopt, report = cand is not None, {}
    if cand is not None and len(exam):
        train_mask = np.ones(len(S), bool)
        train_mask[exam] = False
        yt = y[train_mask]
        if yt.sum() >= 2 and (1 - yt).sum() >= 2:
            challenger = _fit_logreg(X[train_mask], yt, w[train_mask])
            Xe, ye, we = X[exam], y[exam], np.ones(len(exam))
            report = {"exam_size": int(len(exam)),
                      "rules_only": round(_decide_acc(None, Xe, ye, we), 3),
                      "new": round(_decide_acc(challenger, Xe, ye, we), 3)}
            bar = report["rules_only"] + 0.01                  # 必须比"只用规则"好，学习才有意义
            if old.get("keep"):
                report["current"] = round(_decide_acc(old["keep"], Xe, ye, we), 3)
                bar = max(bar, report["current"] - 0.005)
            adopt = report["new"] >= bar
    if adopt and cand is not None:
        model["keep"] = cand
        model["version"] = old.get("version", 0) + 1
    elif old.get("keep") and report.get("current", 1) > report.get("rules_only", 0):
        model["keep"] = old["keep"]                           # 新版本没考过：继续用旧版
    elif old.get("keep"):
        model["paused"] = True                                # 旧版也不如规则了：暂停学习，只用规则
    model["exam"] = report
    model["adopted"] = bool(adopt and cand is not None)
    # 图标 / 图片的类别：只有你真的改过类别（至少 3 次）才学，否则全是程序自己的判断，学了也没用
    kept = [s for s in S if s["keep"]]
    yk = np.array([1.0 if s["kind"] == "图片" else 0.0 for s in kept])
    switches = sum(1 for s in kept if s.get("auto_kind") and s["auto_kind"] != s["kind"])
    model["kind_switches"] = switches
    if switches >= 3 and len(kept) >= 6 and 2 <= yk.sum() <= len(kept) - 2:
        model["kind"] = _fit_logreg(np.array([s["f"] for s in kept], float), yk, np.ones(len(kept)))
    with _LOCK:
        _save_json("model.json", model)
        os.makedirs(LEARN_DIR, exist_ok=True)
        with open(_path("evolution.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": model["t"], "version": model["version"], "adopted": model["adopted"],
                                "n": len(S), **report}, ensure_ascii=False) + "\n")
        _CACHE.clear()
    return model


def record_implicit(img_bgr, page, text_mask=None):
    """下载时这一页没改：当作"自动结果你认可了"，记成弱一点的教训（权重 0.3，不进考卷）。
    同一张图只记一次，免得反复下载把它的分量堆高。"""
    if not enabled():
        return 0
    key = image_key(img_bgr)
    seen = _load_json("implicit_seen.json", {})
    if key in seen or key in _load_json("pages.json", {}):
        return 0
    rows = []
    for r in page.get("icon", []) + page.get("picture", []):
        rows.append({"f": box_features(img_bgr, r[:4], text_mask, r[4] if len(r) > 4 else 60), "keep": 1,
                     "kind": "图片" if r in page.get("picture", []) else "图标", "w": 0.3, "src": "implicit"})
    for r in page.get("rejected", []):
        rows.append({"f": box_features(img_bgr, r[:4], text_mask, r[4] if len(r) > 4 else 40), "keep": 0,
                     "kind": "图标", "w": 0.2, "src": "implicit"})
    if not rows:
        return 0
    now = time.time()
    with _LOCK:
        os.makedirs(LEARN_DIR, exist_ok=True)
        with open(_path("samples.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                r["t"], r["key"] = now, key
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        seen[key] = now
        _save_json("implicit_seen.json", seen)
    return len(rows)


def _model():
    if "model" not in _CACHE:
        _CACHE["model"] = _load_json("model.json", {})
        S = _samples()
        _CACHE["X"] = np.array([s["f"] for s in S], float) if S else np.zeros((0, len(FEATURE_NAMES)))
        _CACHE["y"] = np.array([s["keep"] for s in S], float) if S else np.zeros(0)
    return _CACHE["model"]


def _knn(f, k=7):
    X, y = _CACHE.get("X"), _CACHE.get("y")
    m = _CACHE.get("model", {})
    if X is None or len(X) < k or "keep" not in m:
        return None
    sd = np.asarray(m["keep"]["sd"])
    d = np.linalg.norm((X - np.asarray(f)) / sd, axis=1)
    idx = np.argsort(d)[:k]
    wts = 1 / (d[idx] + 0.3)
    return float((wts * y[idx]).sum() / wts.sum()), float(d[idx[0]])


# ----------------------------------------------------------------------------
# 用学到的东西
# ----------------------------------------------------------------------------
def remembered_page(img_bgr):
    """这张图以前改过：返回上次的修改 {"elements": [...], "texts": [...]}，没有就是 None。"""
    if not enabled():
        return None
    return _load_json("pages.json", {}).get(image_key(img_bgr))


def decide_elements(img_bgr, icons, pics, rejected, text_mask=None):
    """对自动识别的候选框做第二次判断。返回 (edits, notes)：
    edits 是 process_image 能直接用的"想要的元素清单"（和人工修改同一个格式）；没学够、或者学到的结论和规则一致时返回 None。
    notes 是给人看的说明，比如 ["按以前的修改：去掉 2 个、补上 1 个"]。"""
    if not enabled():
        return None, []
    m = _model()
    if "keep" not in m or m.get("n_keep", 0) < 4 or m.get("n_drop", 0) < 4 \
            or m.get("n_explicit", m.get("n", 0)) < MIN_SAMPLES or m.get("version", 1) < 1:
        return None, []
    def judge(el, score):
        f = box_features(img_bgr, el.box, text_mask, score)
        p = float(_predict(m["keep"], [f])[0])
        kn = _knn(f)
        if kn is not None and kn[1] < 3.0:                     # 以前见过很像的：近邻投票也算一半
            p = 0.5 * p + 0.5 * kn[0]
        elif kn is not None:                                   # 离见过的样子很远：往规则分靠
            p = 0.5 * p + 0.5 * (score / 100.0)
        kind = el.kind
        if "kind" in m:
            pk = float(_predict(m["kind"], [f])[0])
            if el.kind == "图标" and pk > 0.8:
                kind = "图片"
            elif el.kind == "图片" and pk < 0.2:
                kind = "图标"
        return p, kind
    edits, dropped, added, swapped = [], 0, 0, 0
    for el in list(pics) + list(icons):
        combined, kind = judge(el, el.score)
        if combined < 0.4:
            dropped += 1
            continue
        if kind != el.kind:
            swapped += 1
            edits.append({"box": list(el.box), "kind": kind, "src": None})
        else:
            edits.append({"box": list(el.box), "kind": el.kind, "src": list(el.box)})
    for el in rejected:
        combined, kind = judge(el, el.score)
        if combined >= 0.65:
            added += 1
            edits.append({"box": list(el.box), "kind": kind, "src": None})
    if not (dropped or added or swapped):
        return None, []
    parts = [s for s in (f"去掉 {dropped} 个" if dropped else "", f"补上 {added} 个" if added else "",
                         f"改了 {swapped} 个类别" if swapped else "") if s]
    return edits, ["按以前的修改学到的：" + "、".join(parts)]


def fix_text(lines):
    """用错字对照表改字：同一对错字至少出现两次，或者改完能组成常用词，才自动改。返回改了几处。"""
    if not enabled():
        return 0
    if "fixes" not in _CACHE:
        table = _load_json("text_fixes.json", {})
        _CACHE["fixes"] = sorted(((k.split("\t")[0], k.split("\t")[1], c) for k, c in table.items()),
                                 key=lambda t: -len(t[0]))
    fixes = _CACHE["fixes"]
    if not fixes:
        return 0
    try:
        import img2pptx as core
        words = core._words()
    except Exception:
        words = {}
    n = 0
    for ln in lines:
        t = ln.text
        for wrong, right, count in fixes:
            if wrong not in t or len(wrong) != len(right) and getattr(ln, "chars", None):
                continue
            ok = count >= 2 or any(right[i:j] in words for i in range(len(right)) for j in range(i + 2, len(right) + 1))
            if ok:
                t = t.replace(wrong, right)
        if t != ln.text:
            if len(t) != len(ln.text):
                ln.chars = None
            ln.text = t
            n += 1
    return n


def stats():
    """给网页显示：学了多少、最近的自动结果有多少不用改。"""
    if not enabled():
        return {"enabled": False}
    m = _load_json("model.json", {})
    hist = []
    try:
        with open(_path("history.jsonl"), encoding="utf-8") as f:
            hist = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        pass
    def rate(items):
        tot = sum(h["total"] for h in items)
        ch = sum(h["changed"] for h in items)
        return None if tot == 0 else round(1 - ch / tot, 3)
    g = [h for h in hist if h["kind"] == "图形"]
    fixes = _load_json("text_fixes.json", {})
    return {"enabled": True, "samples": m.get("n", 0), "corrections": len(hist), "text_fixes": len(fixes),
            "active": "keep" in m and m.get("n_explicit", m.get("n", 0)) >= MIN_SAMPLES and m.get("n_keep", 0) >= 4
                      and m.get("n_drop", 0) >= 4 and m.get("version", 1) >= 1,
            "self_test": m.get("self_test"), "first_rate": rate(g[:10]), "recent_rate": rate(g[-10:]),
            "version": m.get("version", 0), "exam": m.get("exam", {}), "adopted": m.get("adopted", False), "implicit_pages": len(_load_json("implicit_seen.json", {})),
            "pages": len(_load_json("pages.json", {}))}


def forget_page(img_bgr):
    """"撤销这一页的全部修改"：同一张图以后不再自动套用上次的修改（已经学到的图形教训保留）。"""
    if not enabled():
        return
    with _LOCK:
        pages = _load_json("pages.json", {})
        if pages.pop(image_key(img_bgr), None) is not None:
            _save_json("pages.json", pages)


def record_self(img_bgr, log, text_mask=None):
    """自检自修发现并验证过的问题，也记成教训（不需要人）：
      · 回渲染比对改正的错字 → 进错字对照表
      · 补抠的漏掉图标 → 记成"这类东西该抠"（权重 0.5，不进考卷：考卷只用你亲手改的）"""
    if not enabled() or not log:
        return 0
    n = 0
    pairs = log.get("text") or []
    if pairs:
        table = _load_json("text_fixes.json", {})
        for old, new in pairs:
            for i, (a, b) in enumerate(zip(old, new)):
                if a != b:
                    lo, hi = max(0, i - 1), min(len(old), i + 2)
                    k = old[lo:hi] + "\t" + new[lo:hi]
                    table[k] = table.get(k, 0) + 1
                    n += 1
        with _LOCK:
            _save_json("text_fixes.json", table)
            _CACHE.clear()
    boxes = log.get("added") or []
    if boxes:
        now, key = time.time(), image_key(img_bgr)
        with _LOCK:
            os.makedirs(LEARN_DIR, exist_ok=True)
            with open(_path("samples.jsonl"), "a", encoding="utf-8") as f:
                for b in boxes:
                    f.write(json.dumps({"f": box_features(img_bgr, b, text_mask, 40), "keep": 1, "kind": "图标",
                                        "w": 0.5, "src": "self", "t": now, "key": key}, ensure_ascii=False) + "\n")
                    n += 1
            _CACHE.clear()
    return n
