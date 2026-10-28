# v2.1.0 · 在线代码实验室与Colab

- 新增在线代码实验室与数学章节内嵌Python运行器，8个可编辑数学示例覆盖NumPy、SciPy、SymPy和Matplotlib。
- 支持运行、停止、恢复示例、代码下载、文本/异常输出和PNG图像；使用独立Worker、执行超时和输出限制，首次按需加载Pyodide 0.27.7。
- 新增7份中文Notebook：数学、PyTorch基础、TensorFlow基础、CNN、VAE、DDPM、DINOv2，接入Colab按钮和文件下载。
- 模型Notebook固定课程代码提交并验证SHA256；7份Notebook的39个代码单元在本地Jupyter按默认设置逐格执行通过，公开验证摘要。
- 首页、数学章节、完整实验和总览增加运行入口；新增Fork配置，维护者可让Colab指向自己的仓库。
- 验证真实浏览器中的8个示例、绘图、错误提示、停止/重跑、下载内容和390px手机布局；全站12页与176处公式保留。
- 新增Notebook格式/语法/干净输出/下载与启动映射检查，加入GitHub Actions。

运行边界：浏览器Python处理数学小实验；原生torch/tensorflow运行于Colab或本地Python。Pyodide首次需联网；Colab需要读者账号，资源及GPU可用性由Google决定。本地逐格验证不等于在读者的Colab环境逐格验证。

---

# v2.0.0 · 从学习清单到可验证教程

- 新增数学教学：ELBO/Jensen、对角高斯KL、DDPM闭式前向与后验配方、可训练QKV、ODE/SDE、OT/指派、InfoNCE与DINO评价。
- 新增四项完整小规模实验：FashionMNIST CNN、VAE、DDPM训练与反向采样、官方DINOv2冻结线性评价，并提供同架构随机骨干对照。
- 四项实验均保留seed7/17/27真实记录、配置与划分哈希；另提供一次锁定CNN测试（74.71%）。公开结果图与失败分析，不分发数据集或预训练权重。
- 新增26道有解答习题、4项编程练习、参考实现、分层学习路线与100分证据量表；补齐SDE/OT/InfoNCE等系统课程映射。
- 修正基础脚本的样本数分母和教程运行路径；支持生成实验/CNN检查点恢复，明确probe不支持续训。
- 全站11页、176处数学表达式；验证内部链接、图片、表格标签、下载属性和公式编译。10项数值/训练生命周期测试，GitHub Actions增加CPU检查。
- 浏览器实际检查数学排版、实验图片和390px窄屏表格；KaTeX和字体随项目提供，支持离线公式。

在线阅读：https://charlie-sys.github.io/vision-learning-hub/

评价边界：教学规模实验不等于复现论文大规模结果；DINO使用官方预训练模型，未从头训练；不保证任何读者自动取得100分，量表需要实际提交证据。
