#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py —— img2pptx 的本地网页界面

启动：  python app.py        然后浏览器打开 http://127.0.0.1:8765
只用 Python 标准库起服务，转换逻辑全部来自同目录的 img2pptx.py。
默认只监听本机（127.0.0.1），别的电脑访问不到；部署成公开网页时设 IMG2PPTX_PUBLIC=1（见 Dockerfile）。
"""
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

import img2pptx as core
import learning
import page_check

# 公开模式（部署到服务器给别人用）：设环境变量 IMG2PPTX_PUBLIC=1。
# 和本机模式的区别：监听所有网卡、端口取 PORT 环境变量、不能"关闭程序"、每批页数有上限、上传的文件 1 小时后自动删除
PUBLIC = os.environ.get("IMG2PPTX_PUBLIC") == "1"
HOST = "0.0.0.0" if PUBLIC else "127.0.0.1"
PORT = int(os.environ.get("PORT", "7860" if PUBLIC else "8765"))
MAX_PAGES = int(os.environ.get("IMG2PPTX_MAX_PAGES", "16")) if PUBLIC else 10 ** 6
KEEP_SECONDS = 3600                                               # 公开模式下任务保留多久
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(tempfile.gettempdir(), f"img2pptx_web_{PORT}")  # 上传的图和生成的文件放这里，重启时清空（按端口分开，本机版和在线版互不影响）
MAX_UPLOAD = 40 * 1024 * 1024                                     # 单张图片上限 40 MB
ALLOWED = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

JOBS = {}                       # 任务编号 -> 任务信息
RUN_LOCK = threading.Lock()     # 模型不适合同时跑多个任务，排队执行
WAITING = []                    # 排队等着的任务编号（按先后）


def janitor():
    """公开模式：每 5 分钟清一次，删掉 1 小时没动静的任务和它的文件（不动正在跑或排队的）。"""
    while True:
        time.sleep(300)
        now = time.time()
        for jid, job in list(JOBS.items()):
            if job["state"] in ("running", "queued") or now - job.get("touched", now) < KEEP_SECONDS:
                continue
            JOBS.pop(jid, None)
            shutil.rmtree(job["dir"], ignore_errors=True)


def new_job(opts):
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(WORK, jid)
    os.makedirs(d)
    JOBS[jid] = {"id": jid, "dir": d, "files": [], "names": [], "state": "uploading", "done": 0, "pages": [],
                 "error": None, "live": None, "opts": opts, "output": None, "started": None, "seconds": None,
                 "touched": time.time()}
    return JOBS[jid]


def run_job(job, only=None):
    """后台线程：逐张转换，每完成一张就更新进度。
    only：只重新处理这几页（人工调整之后），其余页沿用上次的结果。"""
    WAITING.append(job["id"])
    try:
        RUN_LOCK.acquire()
    finally:
        if job["id"] in WAITING:
            WAITING.remove(job["id"])
    try:
        _run_job(job, only)
    finally:
        job["touched"] = time.time()
        RUN_LOCK.release()


def _run_job(job, only=None):
    job["state"], job["started"], job["done"] = "running", time.time(), 0
    job["warnings"] = []
    try:
        n = len(job["files"])
        job.setdefault("results", [None] * n)
        job.setdefault("ocr", [None] * n)
        job.setdefault("edits", [None] * n)
        job.setdefault("textedits", [[] for _ in range(n)])
        job.setdefault("rejected", [[] for _ in range(n)])
        for i, path in enumerate(job["files"]):
            if only is not None and i not in only and job["results"][i] is not None:
                job["done"] = i + 1
                continue
            # live：当前这一页识别到哪一步了、已经找到哪些框。页面靠它一步步把框画出来
            job["live"] = {"page": i, "stage": "ocr", "text": [], "picture": [], "icon": [], "rejected": []}
            def on_stage(name, data, live=job["live"]):
                if name != "rejected":                      # "rejected" 只是附带数据，不算一个阶段
                    live["stage"] = name
                if data is not None:
                    live[name] = data
            try:
                job["results"][i] = core.process_image(
                    path, "auto", "chi_sim+eng",
                    with_icons=job["opts"]["icons"], with_pictures=job["opts"]["pictures"],
                    inpaint=job["opts"]["inpaint"], on_stage=on_stage,
                    ocr=job["ocr"][i], edits=job["edits"][i], text_edits=job["textedits"][i])
            except Exception:
                # 抠图那一步碰到没见过的版面出错时，不让整批失败：这一页退回"只转文字"，其余照常
                traceback.print_exc()
                job["warnings"].append(f"第 {i + 1} 页（{job['names'][i]}）识别图标/图片时出错，这一页只转换了文字")
                job["results"][i] = core.process_image(
                    path, "auto", "chi_sim+eng", with_icons=False, with_pictures=False,
                    inpaint=job["opts"]["inpaint"], on_stage=on_stage, ocr=job["ocr"][i],
                    text_edits=job["textedits"][i])
            job.setdefault("notes", [[] for _ in range(n)])[i] = list(getattr(core.process_image, "last_notes", []))
            job["ocr"][i] = core.process_image.last_ocr      # 文字识别最费时间，留着，人工调整后重新生成时不用再做
            job["rejected"][i] = list(getattr(core.process_image, "last_rejected", []))
            job["done"] = i + 1
        prs, pages = None, []
        for i in range(n):
            clean, blocks, icons, W, H = job["results"][i]
            if prs is None:
                prs = core.Presentation()
                prs.slide_width = core.SLIDE_W_EMU
                prs.slide_height = int(core.SLIDE_W_EMU * H / W)
            core.add_slide(prs, clean, blocks, icons, W, H, core.default_ppt_font())
            cv2.imencode(".png", clean)[1].tofile(os.path.join(job["dir"], f"clean_{i}.png"))
            # 页面上要画的三类框：文字 / 图标 / 图片
            texts = []
            for b in blocks:
                x0 = min(l.ink[0] for l in b.lines)
                y0 = min(l.ink[1] for l in b.lines)
                x1 = max(l.ink[2] for l in b.lines)
                y1 = max(l.ink[3] for l in b.lines)
                texts.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0),
                              int(round(100 * min(l.score for l in b.lines))), "\n".join(l.text for l in b.lines)])
            def row(ic):                                    # 最后一项：是不是人工调整出来的
                return [int(v) for v in ic.box] + [ic.score, ic.why, 1 if ic.why.startswith("手动") else 0]
            lines = [[int(l.ink[0]), int(l.ink[1]), int(l.ink[2] - l.ink[0]), int(l.ink[3] - l.ink[1]), l.text,
                      1 if getattr(l, "manual", False) else 0]
                     for b in blocks for l in b.lines if l.ink]  # 逐行的框和文字：手动改字用
            pages.append({
                "name": job["names"][i], "w": W, "h": H,
                "text": texts, "lines": lines,
                "icon": [row(ic) for ic in icons if ic.kind == "图标"],
                "picture": [row(ic) for ic in icons if ic.kind == "图片"],
                "rejected": [row(ic) for ic in job["rejected"][i]],
                "edited": job["edits"][i] is not None or bool(job["textedits"][i]),
                "notes": job["notes"][i] if i < len(job.get("notes", [])) else [],
                "issues": [],
            })
            try:                                            # 自检只负责报告：出错也不影响转换结果
                src = cv2.imdecode(np.fromfile(job["files"][i], dtype=np.uint8), cv2.IMREAD_COLOR)
                p = pages[-1]
                p["issues"], heat = page_check.check_page(src, clean, p["text"], p["icon"], p["picture"], p["lines"], icons)
                cv2.imencode(".jpg", heat)[1].tofile(os.path.join(job["dir"], f"heat_{i}.jpg"))
            except Exception:
                traceback.print_exc()
        job["pages"] = pages
        stem = os.path.splitext(job["names"][0])[0]
        name = stem + (f"_等{n}页" if n > 1 else "") + ".pptx"
        out = os.path.join(job["dir"], "out.pptx")
        prs.save(out)
        job["output"], job["output_name"] = out, name
        job["seconds"] = round(time.time() - job["started"], 1)
        job["state"] = "done"
    except Exception as e:                                  # 出错时把原因带回页面，而不是让页面一直转圈
        traceback.print_exc()
        job["error"] = f"{type(e).__name__}: {e}"
        job["state"] = "error"


def learn_from_download(job):
    """下载了、但这一页没改：当作"自动结果你认可了"，记成弱一点的教训；攒够了再训练（训练会先考试再上岗）。"""
    n = 0
    for i, page in enumerate(job.get("pages") or []):
        if job["edits"][i] is not None or job["textedits"][i] or page.get("notes"):
            continue                                            # 改过的、按以前修改生成的：不算"认可自动结果"
        try:
            src = cv2.imdecode(np.fromfile(job["files"][i], dtype=np.uint8), cv2.IMREAD_COLOR)
            tmask = np.zeros(src.shape[:2], np.uint8)
            for r in page.get("lines", []):
                x, y, w, h = [int(v) for v in r[:4]]
                tmask[max(0, y):y + h, max(0, x):x + w] = 255
            n += learning.record_implicit(src, page, tmask)
        except Exception:
            traceback.print_exc()
    if n:
        try:
            learning.train()
        except Exception:
            traceback.print_exc()


def public(job):
    keys = ("id", "state", "done", "pages", "error", "seconds", "live")
    out = {k: job[k] for k in keys}
    out["total"] = len(job["files"])
    out["names"] = job["names"]
    out["download_name"] = job.get("output_name")
    out["warnings"] = job.get("warnings", [])
    out["ahead"] = WAITING.index(job["id"]) + (1 if RUN_LOCK.locked() else 0) if job["id"] in WAITING else 0
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):                          # 只打印出错的请求，终端保持干净
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("  %s\n" % (fmt % args))

    # ---- 回应的几种形式 ----
    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype, download_name=None):
        if not os.path.isfile(path):
            return self.send_json({"error": "文件不存在"}, 404)
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(download_name))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def job_or_404(self, jid):
        job = JOBS.get(jid)
        if job:
            job["touched"] = time.time()
        if not job:
            self.send_json({"error": "找不到这个任务，可能服务重启过，请重新上传"}, 404)
        return job

    # ---- 路由 ----
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send_file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
        if path == "/api/info":
            lama = any(os.path.exists(p) for p in core.LAMA_PATHS)
            try:
                import rapidocr_onnxruntime  # noqa: F401
                ocr = "rapidocr"
            except ImportError:
                ocr = "tesseract"
            return self.send_json({"lama": lama, "ocr": ocr, "public": PUBLIC, "max_pages": MAX_PAGES if PUBLIC else None,
                                   "learning": learning.stats()})
        m = re.fullmatch(r"/api/jobs/(\w+)", path)
        if m:
            job = self.job_or_404(m.group(1))
            return job and self.send_json(public(job))
        m = re.fullmatch(r"/api/jobs/(\w+)/image/(\d+)/(orig|clean|heat)", path)
        if m:
            job = self.job_or_404(m.group(1))
            if not job:
                return
            i = int(m.group(2))
            if i >= len(job["files"]):
                return self.send_json({"error": "没有这一页"}, 404)
            if m.group(3) == "clean":
                return self.send_file(os.path.join(job["dir"], f"clean_{i}.png"), "image/png")
            if m.group(3) == "heat":
                return self.send_file(os.path.join(job["dir"], f"heat_{i}.jpg"), "image/jpeg")
            return self.send_file(os.path.join(job["dir"], f"orig_{i}.png"), "image/png")
        m = re.fullmatch(r"/api/jobs/(\w+)/download", path)
        if m:
            job = self.job_or_404(m.group(1))
            if not job:
                return
            if job["state"] != "done":
                return self.send_json({"error": "还没转换完"}, 409)
            threading.Thread(target=learn_from_download, args=(job,), daemon=True).start()
            return self.send_file(job["output"], "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                  job["output_name"])
        self.send_json({"error": "没有这个地址"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        length = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/jobs":
            self.rfile.read(length)
            opts = {"icons": q.get("icons", ["1"])[0] == "1", "pictures": q.get("pictures", ["1"])[0] == "1",
                    "inpaint": q.get("inpaint", ["auto"])[0] if q.get("inpaint", ["auto"])[0] in ("auto", "lama", "telea") else "auto"}
            return self.send_json({"id": new_job(opts)["id"]})
        m = re.fullmatch(r"/api/jobs/(\w+)/files", u.path)
        if m:
            job = self.job_or_404(m.group(1))
            if not job:
                self.rfile.read(length)
                return
            name = os.path.basename(q.get("name", ["image.png"])[0]) or "image.png"
            if length > MAX_UPLOAD:
                self.rfile.read(length)
                return self.send_json({"error": f"{name} 超过 40 MB，太大了"}, 413)
            data = self.rfile.read(length)
            if not name.lower().endswith(ALLOWED):
                return self.send_json({"error": f"{name} 不是支持的图片格式（PNG / JPG / WebP / BMP）"}, 400)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return self.send_json({"error": f"{name} 打不开，文件可能损坏"}, 400)
            stem = os.path.splitext(name)[0]
            parts = core.split_contact_sheet(img)              # 一张图里拼着好几页：拆成单页
            items = [(img, name)] if not parts else [(pp, f"{stem}_第{k + 1}页.png") for k, pp in enumerate(parts)]
            if len(job["files"]) + len(items) > MAX_PAGES:
                return self.send_json({"error": f"在线版每次最多转换 {MAX_PAGES} 张，{name} 没有加入"}, 413)
            first = len(job["files"])
            for pic, nm in items:
                i = len(job["files"])
                # 统一存成 PNG：既给转换用，也给页面预览用（浏览器不一定能直接显示所有格式）
                src = os.path.join(job["dir"], f"{i:03d}_{os.path.splitext(nm)[0]}.png")
                cv2.imencode(".png", pic)[1].tofile(src)
                shutil.copyfile(src, os.path.join(job["dir"], f"orig_{i}.png"))
                job["files"].append(src)
                job["names"].append(nm)
            return self.send_json({"index": first, "pages": len(items), "w": int(img.shape[1]), "h": int(img.shape[0])})
        m = re.fullmatch(r"/api/jobs/(\w+)/start", u.path)
        if m:
            self.rfile.read(length)
            job = self.job_or_404(m.group(1))
            if not job:
                return
            if not job["files"]:
                return self.send_json({"error": "还没有上传图片"}, 400)
            if job["state"] == "uploading":
                job["state"] = "queued"
                threading.Thread(target=run_job, args=(job,), daemon=True).start()
            return self.send_json(public(job))
        m = re.fullmatch(r"/api/jobs/(\w+)/redo", u.path)      # 人工调整后重新生成
        if m:
            body = self.rfile.read(length)
            job = self.job_or_404(m.group(1))
            if not job:
                return
            if job["state"] not in ("done", "error") or not job.get("results"):
                return self.send_json({"error": "上一次转换还没结束"}, 409)
            try:
                req = json.loads(body.decode("utf-8"))["pages"]
                only, lessons = [], []
                for k, v in req.items():
                    i = int(k)
                    if not 0 <= i < len(job["files"]):
                        raise ValueError("页码不对")
                    if v is None:                               # 恢复自动识别：图形和文字的修改都清掉
                        job["edits"][i], job["textedits"][i] = None, []
                        job.setdefault("forget", []).append(i)
                        only.append(i)
                        continue
                    if isinstance(v, list):                     # 旧格式：只有图形清单
                        v = {"elements": v}
                    lst = v.get("elements")
                    if lst is not None:
                        job["edits"][i] = [{"box": [float(x) for x in e["box"]][:4],
                                            "kind": "图片" if e.get("kind") == "图片" else "图标",
                                            "src": [int(x) for x in e["src"]][:4] if e.get("src") else None} for e in lst]
                    for e in v.get("texts") or []:             # 改字：累加在之前的修改后面，按顺序生效
                        t = e.get("text")
                        job["textedits"][i].append({
                            "box": [float(x) for x in e["box"]][:4] if e.get("box") else None,
                            "src": [float(x) for x in e["src"]][:4] if e.get("src") else None,
                            "text": None if t is None else str(t)[:500]})
                    only.append(i)
                    lessons.append((i, job["edits"][i] if lst is not None else None,
                                    job["textedits"][i][len(job["textedits"][i]) - len(v.get("texts") or []):]))
            except Exception as e:
                return self.send_json({"error": f"调整内容格式不对：{e}"}, 400)
            pages_before = list(job.get("pages") or [])
            def learn(lessons=lessons, pages_before=pages_before, forget=job.pop("forget", [])):
                """把这次的修改记成教训（在后台做，不耽误重新生成）"""
                for i, elements, texts in lessons:
                    try:
                        if i >= len(pages_before):
                            continue
                        src = cv2.imdecode(np.fromfile(job["files"][i], dtype=np.uint8), cv2.IMREAD_COLOR)
                        tmask = np.zeros(src.shape[:2], np.uint8)
                        for r in pages_before[i].get("lines", []):
                            x, y, w, h = [int(v) for v in r[:4]]
                            tmask[max(0, y):y + h, max(0, x):x + w] = 255
                        learning.record_page(src, pages_before[i], elements, texts, tmask)
                    except Exception:
                        traceback.print_exc()
                for i in forget:
                    try:
                        learning.forget_page(cv2.imdecode(np.fromfile(job["files"][i], dtype=np.uint8), cv2.IMREAD_COLOR))
                    except Exception:
                        traceback.print_exc()
            threading.Thread(target=learn, daemon=True).start()
            job["state"], job["error"] = "queued", None
            threading.Thread(target=run_job, args=(job, only), daemon=True).start()
            return self.send_json(public(job))
        if u.path == "/api/quit" and not PUBLIC:                # 页面上的"关闭程序"（公开模式下没有）
            self.rfile.read(length)
            self.send_json({"ok": True})
            threading.Timer(0.4, lambda: os._exit(0)).start()
            return
        self.rfile.read(length)
        self.send_json({"error": "没有这个地址"}, 404)


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        print(f"端口 {PORT} 已被占用 —— 多半是已经开着一个了，直接打开 http://{HOST}:{PORT} 就行。")
        webbrowser.open(f"http://{HOST}:{PORT}")
        return
    url = f"http://{HOST}:{PORT}"
    print(f"img2pptx 已启动： {url}" + ("（公开模式）" if PUBLIC else "\n关掉这个窗口（或按 Ctrl+C）就停止。"), flush=True)
    if PUBLIC:
        threading.Thread(target=janitor, daemon=True).start()
    if "--no-browser" not in sys.argv and not PUBLIC:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
