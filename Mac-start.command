#!/bin/bash
# 图片转PPT（Mac）：第一次双击会自动安装运行环境（需要联网，约 3–10 分钟），以后双击直接启动。
cd "$(dirname "$0")"
PY=""
for c in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v $c >/dev/null 2>&1 && $c -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then PY=$c; break; fi
done
if [ ! -x .venv/bin/python ]; then
  if [ -z "$PY" ]; then
    echo "没有找到合适的 Python（需要 3.9 – 3.12）。"
    echo "请到 https://www.python.org/downloads/ 下载安装 Python 3.12，装好后再双击本文件。"
    read -n 1 -s -r -p "按任意键关闭…"; exit 1
  fi
  echo "第一次运行：正在安装运行环境，请保持联网，不要关闭这个窗口…"
  $PY -m venv .venv || { echo "创建运行环境失败。"; read -n 1 -s -r -p "按任意键关闭…"; exit 1; }
  # 电脑开着代理软件时，命令行默认不走代理，会连不上网；这里读取系统代理设置并沿用
  if [ -z "$https_proxy" ] && scutil --proxy 2>/dev/null | grep -q "HTTPSEnable : 1"; then
    PH=$(scutil --proxy | awk '/HTTPSProxy/ {print $3}'); PP=$(scutil --proxy | awk '/HTTPSPort/ {print $3}')
    [ -n "$PH" ] && [ -n "$PP" ] && export https_proxy="http://$PH:$PP" http_proxy="http://$PH:$PP"
  fi
  # 依次试几个下载源：有的网络连不上国内镜像，有的连不上官方源
  OK=0
  for SRC in https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.aliyun.com/pypi/simple https://pypi.org/simple; do
    echo "从 $SRC 下载…"
    if .venv/bin/python -m pip install --retries 1 --timeout 20 -r requirements.txt -i $SRC; then OK=1; break; fi
  done
  if [ $OK -ne 1 ]; then
    rm -rf .venv
    echo; echo "安装失败（多半是网络问题）。检查网络后再双击一次。"
    read -n 1 -s -r -p "按任意键关闭…"; exit 1
  fi
  echo "安装完成。"
fi
if [ ! -f models/lama_fp32.onnx ]; then
  echo "正在下载背景补全模型（约 200 MB，只需一次）…"
  mkdir -p models
  for U in https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx; do
    if curl -L --fail -o models/lama_fp32.onnx.part "$U"; then mv models/lama_fp32.onnx.part models/lama_fp32.onnx; break; fi
  done
  [ -f models/lama_fp32.onnx ] || echo "模型没下载下来：程序仍可使用，但会改用快速算法，复杂背景会糊。可稍后手动下载放进 models 文件夹。"
fi
echo "正在启动，浏览器会自动打开。用完后在网页最下面点“关闭程序”，或直接关掉这个窗口。"
exec .venv/bin/python app.py
