# Vision Learning Hub · 视觉论文学习库

面向扩散模型、视觉 Transformer、VAE 和 DINO 的中文学习项目：30 篇代表性论文、数学/PyTorch/视觉前置教程、76 个逐项学习入口，以及一个无需外部数据集的 PyTorch 实验。

在线阅读：**[https://charlie-sys.github.io/vision-learning-hub/](https://charlie-sys.github.io/vision-learning-hub/)** · [Fork 仓库](https://github.com/charlie-sys/vision-learning-hub/fork)

## 阅读入口

- [可搜索论文目录](docs/index.html)
- [前置知识完整教程](docs/tutorial.html) · [Markdown 源文](content/tutorial.md)
- [论文阅读完整指南](docs/guide.html) · [Markdown 源文](content/guide.md)
- [PyTorch 小实验网页](docs/lab.html) · [Python 源码](examples/vision_basics_lab.py)
- [英文论文翻译阅读指南](docs/translation.html)：目录中每篇论文提供“翻译阅读”弹窗
- [论文清单 CSV](docs/papers.csv) · [BibTeX](data/references.bib)

HTML 已随仓库提供。下载仓库后可直接用浏览器打开 `docs/index.html`，也可以在项目根目录运行 `python -m http.server 8000 --directory docs`，再访问 http://localhost:8000 。GitHub 文件页会展示 HTML 源码，在线渲染请使用仓库的 GitHub Pages 地址。

教程公式使用 LaTeX 编写，由项目自带的 KaTeX 脚本、样式和字体渲染；离线打开 HTML 也能显示数学排版。长公式在自己的区域内横向滚动。

## Fork、运行和修改

点击 GitHub 的 **Fork** 创建自己的副本，再 Clone 到本地。网页阅读无需安装依赖。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build.py
python scripts/validate.py
```

实验需要另行按 [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/)安装适合本机的 PyTorch，然后运行：

```bash
python examples/vision_basics_lab.py
```

## 获取论文全文与离线阅读

目录的 PDF 按钮可直接打开原始公开全文。也可下载到本地；共 30 篇，可能需要较长时间和数百 MB 空间：

```bash
python scripts/download_papers.py --list
python scripts/download_papers.py --ids 1,2,15
python scripts/download_papers.py
python scripts/build.py --local-pdfs
```

最后一条命令把已下载论文接入本地网页；未下载的仍指向原始来源。若某个来源失效，程序会报告题名和来源，可从原论文页面继续查找。论文 PDF 使用各自许可，详见 [第三方资料说明](THIRD_PARTY_NOTICES.md)。

## 翻译阅读

点击论文卡片的“翻译阅读”可打开当前论文的翻译助手，提供原始 PDF、链接复制、沉浸式翻译扩展和 BabelDOC 在线入口。扩展方式适合边读边看双语；在线文档方式需要由读者在第三方页面上传 PDF 并启动任务。当前静态网站提供工具入口和指引，实际翻译由工具方执行，不提供本站翻译后端。使用条件与额度见各工具当前页面。

## 项目结构

```text
content/       可编辑的教程与指南 Markdown
data/          论文元数据与引用
examples/      可运行的 PyTorch 小实验
templates/     网页模板与响应式排版
scripts/       构建、验证、下载和打包脚本
docs/          已生成的完整静态网页，可用于 GitHub Pages
```

## 发布自己的网页

Fork 后在仓库 Settings → Pages 中选择从分支部署，分支 `main`、目录 `/docs`。修改源文件后重新构建并提交 `docs/`。验证工作流会检查 HTML 内部链接与表格结构。

## 内容范围

这是一份以经典方法和代表性演进为主的学习清单，涵盖 2013–2025 年论文，不是穷尽综述或 2026 年前沿排名。小实验的合成分类任务用于理解训练流程，不能代表真实视觉任务性能。欢迎通过 Issue/PR 修正和扩充。

原创内容与代码采用 [MIT 许可证](LICENSE)。第三方论文与课程遵循各自许可。
