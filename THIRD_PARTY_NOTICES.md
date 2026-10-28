# 第三方资料说明

原创教程、网页模板和实验/构建代码使用根目录 MIT 许可证。论文、课程、官方文档和链接目标的版权及许可归各自权利人所有；MIT 许可证不覆盖它们。

论文记录及引用见 `data/papers.json`、`data/references.bib`。仓库中的论文目录链接到 arXiv、会议网站或作者公开页面；下载脚本直接从这些公开来源获取文件。公开访问不等于允许再分发，因此第三方 PDF 不纳入本仓库的 Git 历史。下载后请遵循原始来源的许可。

学习链接主要来自《动手学深度学习》、PyTorch/TorchVision、MIT OpenCourseWare、斯坦福 CS231n/CS236、NumPy、Python、Pillow 和 scikit-learn。教程包含资料入口及学习目标；外部课程内容通过原始页面访问。

数学排版使用 KaTeX 0.16.22（MIT 许可证），脚本、样式与字体存放于 `vendor/katex/`，完整许可证见该目录的 `LICENSE`。构建时复制到 `docs/assets/katex/`，支持离线渲染。来源：https://github.com/KaTeX/KaTeX 。

四项实验使用FashionMNIST（Han Xiao、Kashif Rasul、Roland Vollgraf；Zalando Research，MIT许可），数据和许可见 https://github.com/zalandoresearch/fashion-mnist 。`evidence/` 中的重建示例含该数据集图像，生成图与指标为本项目教学运行结果；原始完整数据集不随仓库分发。

DINOv2实验从Meta官方仓库与服务器加载标准ViT-S/14模型，固定代码提交为 `e1277af2ba9496fbadf7aec6eba56e8d882d1e35`。原代码与相应标准模型遵循其Apache-2.0许可，详见 https://github.com/facebookresearch/dinov2 及模型卡。仓库不重新分发DINO权重。PyTorch、TorchVision、Pillow等依赖分别遵循各自许可。

新增教学入口来自MIT Diffusion/Flow Matching 2025、UvA Deep Learning Tutorials、Peyré/Cuturi的Computational Optimal Transport、SciPy官方文档。本站提供原创中文解释和链接，不复制这些课程全文或Notebook。

网页Python运行器使用Pyodide 0.27.7（MPL-2.0），由jsDelivr公开CDN在用户点击运行后加载，来源 https://github.com/pyodide/pyodide 。运行时与其依赖包未随项目重新分发；NumPy、SciPy、SymPy、Matplotlib及其传递依赖遵守各自许可。运行器和Notebook教学代码为本项目原创。Colab是Google提供的外部服务，其账号、计算资源与条款由服务方管理；项目不转售或代理其计算资源。
