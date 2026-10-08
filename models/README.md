背景补全模型 `lama_fp32.onnx`（约 200 MB）放在这个文件夹。

启动脚本会自动下载；也可以手动从 https://huggingface.co/Carve/LaMa-ONNX 下载（国内可用 hf-mirror.com）。
没有这个文件程序也能运行，只是改用 OpenCV 的快速算法，复杂背景上抹掉文字后会比较糊。

## words.txt（识别纠错用的常用词表）

`words.txt` 是从 [jieba](https://github.com/fxsjy/jieba)（MIT 许可）的词库里取出的 2–4 字常用词及词频，用于“形近字纠错”：
识别结果里某个字换成和它长得像的字后能组成常用词（如“目我”→“自我”），而原来的字组不成词时，就自动改正。
文件缺失时程序照常运行，只是不做这一步纠错。
