# 在线版：部署到 Hugging Face Spaces / 魔搭 ModelScope 创空间（Docker 类型）或任何能跑 Docker 的服务器。
# 本地自己用不需要这个文件，直接双击 Mac-start.command / Windows-start.bat。
FROM python:3.11-slim

# 中文参考字体（判断宋体 / 黑体、粗细用）+ OpenCV 需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-cjk libglib2.0-0 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 背景补全模型（约 200 MB）：构建镜像时下载一次。国内服务器先试镜像站
RUN mkdir -p models && \
    for U in https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx \
             https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx; do \
        curl -fL --retry 3 -o models/lama_fp32.onnx.part "$U" && mv models/lama_fp32.onnx.part models/lama_fp32.onnx && break; \
    done; \
    ls -l models/ ; \
    python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"   # 预先加载一次，确认文字识别能用

# 托管平台通常以普通用户运行：让临时目录可写
RUN chmod -R a+rwX /app
ENV IMG2PPTX_PUBLIC=1 PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py", "--no-browser"]
