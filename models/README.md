背景补全模型 `lama_fp32.onnx`（约 200 MB）放在这个文件夹。

启动脚本会自动下载；也可以手动从 https://huggingface.co/Carve/LaMa-ONNX 下载（国内可用 hf-mirror.com）。
没有这个文件程序也能运行，只是改用 OpenCV 的快速算法，复杂背景上抹掉文字后会比较糊。
