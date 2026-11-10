# 部署成在线网页

程序本身就是一个网页服务（`app.py`）。设环境变量 `IMG2PPTX_PUBLIC=1` 就是**在线版**，和本机版的区别：

- 监听所有网卡，端口取环境变量 `PORT`（默认 7860）
- 没有“关闭程序”
- 每次最多转换 12 张（`IMG2PPTX_MAX_PAGES` 可改）
- 上传的图片和生成的 PPT 1 小时后自动删除
- 同一时间只跑一个任务，其他人排队，页面上显示前面还有几个任务

仓库里的 `Dockerfile` 已经配好：中文字体、依赖、背景补全模型（构建时自动下载）都会装进镜像。

## 方式一：魔搭 ModelScope 创空间（国内访问快，免费）

1. 注册 / 登录 modelscope.cn，进入“创空间” → “创建创空间”
2. 类型选 **Docker**，可见性选“公开”
3. 把本仓库的文件上传到创空间的代码仓库（网页上传，或 `git push`）
4. 端口填 **7860**，等它构建完成（第一次约 10 分钟，主要是下载模型）
5. 构建好后会得到一个 `modelscope.cn/studios/…` 的网址，发给别人就能用

## 方式二：Hugging Face Spaces（免费，国内需要代理）

1. 登录 huggingface.co → New Space → SDK 选 **Docker**
2. 把本仓库文件推上去，并在 Space 的 `README.md` 最前面加上：

   ```yaml
   ---
   title: 图片转可编辑PPT
   sdk: docker
   app_port: 7860
   ---
   ```

3. 等构建完成，网址是 `huggingface.co/spaces/你的用户名/空间名`

## 方式三：自己的云服务器

```bash
docker build -t img2pptx .
docker run -d -p 80:7860 --restart always img2pptx
```

## 速度和资源

免费的 2 核 CPU 上，每页约 30–60 秒（Mac M 系列芯片上约 25 秒）。内存约需 2–3 GB。
免费空间一段时间没人访问会休眠，下次打开要等它重新启动（约 1 分钟）。
