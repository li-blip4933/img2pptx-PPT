@echo off
chcp 65001 >nul
title 图片转PPT
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run

set "PY="
py -3.12 -c "print(1)" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 -c "print(1)" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.10 -c "print(1)" >nul 2>&1 && set "PY=py -3.10"
if not defined PY py -3.9 -c "print(1)" >nul 2>&1 && set "PY=py -3.9"
if not defined PY python -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo 没有找到合适的 Python（需要 3.9 - 3.12）。
  echo 请到 https://www.python.org/downloads/ 下载安装 Python 3.12，
  echo 安装时务必勾选 "Add python.exe to PATH"，装好后再双击本文件。
  pause
  exit /b 1
)
echo 第一次运行：正在安装运行环境，请保持联网，不要关闭这个窗口（约 3-10 分钟）...
%PY% -m venv .venv
if errorlevel 1 (
  echo 创建运行环境失败。
  pause
  exit /b 1
)
set "OK="
for %%S in (https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.aliyun.com/pypi/simple https://pypi.org/simple) do (
  if not defined OK (
    echo 从 %%S 下载...
    ".venv\Scripts\python.exe" -m pip install --retries 1 --timeout 20 -r requirements.txt -i %%S && set "OK=1"
  )
)
if not defined OK (
  rmdir /s /q .venv
  echo.
  echo 安装失败（多半是网络问题）。检查网络后再双击一次。
  pause
  exit /b 1
)
echo 安装完成。

:run
if not exist "models\lama_fp32.onnx" (
  echo 正在下载背景补全模型（约 200 MB，只需一次）...
  if not exist models mkdir models
  curl -L --fail -o "models\lama_fp32.onnx.part" https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx || curl -L --fail -o "models\lama_fp32.onnx.part" https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx
  if exist "models\lama_fp32.onnx.part" (ren "models\lama_fp32.onnx.part" lama_fp32.onnx) else (echo 模型没下载下来：程序仍可使用，但复杂背景会糊。)
)
echo 正在启动，浏览器会自动打开。用完后在网页最下面点“关闭程序”，或直接关掉这个窗口。
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
