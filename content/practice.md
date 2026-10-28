# 四项完整 PyTorch 视觉实验

从真实数据分类到 VAE、DDPM，再到官方 DINOv2 冻结特征评价。先读[数学教学](math.html)，做[练习](answers.html)，运行后对照[实测记录](evidence.html)。这些是小规模教学实现，目标是解释流程，不承诺论文级指标。

无需本地安装时，直接使用每项实验的Colab按钮；另有[PyTorch / TensorFlow基础Notebook与网页数学实验](playground.html)。下面的命令供本地运行使用。

## 1. 环境与第一次运行

以下命令一律从 Clone 或解压后的**仓库根目录**运行。建议 Python 3.10 或 3.11；本次实测 Python 3.9、PyTorch 2.8、torchvision 0.23，DINOv2 使用固定的兼容历史提交。不同系统的 PyTorch 安装命令应从[官方安装向导](https://pytorch.org/get-started/locally/)选择；先安装与本机匹配的 torch/torchvision，再安装实验依赖。

```bash
python -m pip install -r requirements-labs.txt
python examples/course_labs.py --help
python examples/solutions.py
python -m unittest discover -s tests -v
```

训练代码[course_labs.py](course_labs.py)、[参考解答](solutions.py)、[练习骨架](exercises.py)可单独下载；单文件运行时把 `examples/course_labs.py` 改成该文件的实际路径。为保证文件布局，推荐下载完整项目。`exercises.py` 的 TODO 故意未实现，不属于现成训练脚本。

没有独立显卡可加 `--device cpu`，并先用 `--train-size 256 --val-size 128 --epochs 1`。这会降低运行成本，也会降低结果质量。默认 `auto` 选择 CUDA（可用时），否则 CPU；Apple设备当前同样走CPU，未提供MPS适配。

### 无网络冒烟测试

```bash
python examples/course_labs.py cnn --dataset synthetic --train-size 40 --val-size 20 --epochs 1 --batch-size 10 --device cpu --output runs/smoke-cnn
python examples/course_labs.py vae --dataset synthetic --train-size 40 --val-size 20 --epochs 1 --batch-size 10 --device cpu --output runs/smoke-vae
python examples/course_labs.py ddpm --dataset synthetic --train-size 40 --val-size 20 --epochs 1 --batch-size 10 --steps 8 --device cpu --output runs/smoke-ddpm
python examples/course_labs.py probe --encoder tiny --dataset synthetic --train-size 40 --val-size 20 --epochs 1 --batch-size 10 --device cpu --output runs/smoke-probe
```

它们只验证训练、评价、文件生成和采样能完成。`tiny` 是随机小卷积骨干，**不能用于声称已完成 DINO 表征实验**。

## 2. 数据划分与记录规范

使用 FashionMNIST 10 类 28×28 灰度图。下载源为[数据集作者仓库](https://github.com/zalandoresearch/fashion-mnist)，torchvision 验证原始资源校验和。第一次下载需要网络，之后使用 `data/datasets/` 缓存。数据遵守原始许可，未随项目复制完整数据集。

固定 seed 打乱官方 60000 张训练数据，前 50000 张形成训练池、后 10000 张形成验证池，再按参数截取子集；训练和验证索引不重叠。官方测试集默认不读取。`metrics.json` 保存种子、划分索引哈希、超参数、环境、逐轮指标和耗时；索引哈希可检查是否用了同一划分。

每个实验使用独立输出目录。CNN 按验证准确率、生成模型按验证损失保存 `best.pt`，每轮另存 `last.pt`。`curves.svg` 各面板独立缩放，便于看趋势，不能用视觉高度比较不同指标绝对量级；实际数值见 JSON。权重与数据默认忽略，避免上传大文件。

## 3. 实验一：CNN 真实图像分类

<!-- COLAB:03_cnn -->

**目标：** 掌握 Dataset→batch→logits→交叉熵→反向传播→验证与检查点的完整流程。网络两层卷积和池化，接小型分类头；输出10类 logits。

```bash
python examples/course_labs.py cnn --train-size 2048 --val-size 512 --epochs 3 --batch-size 64 --output runs/verified-cnn
```

训练阶段开启梯度并每步清零，验证阶段 `eval()` 配合 `inference_mode()`。损失按实际样本数加权，准确率用总正确数除总样本数。`metrics.json` 包含10×10混淆矩阵，行为真实类别、列为预测类别。类别顺序见作者数据说明：T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot。

**验收：** 指标超过同验证集上的多数类基线；用混淆矩阵指出至少一对易混类别；检查训练和验证差距。准确率要求来自你的实际基线与设置，不强制每台机器重复同一小数。

**对照练习：** 保持划分与轮数相同，将学习率从 `0.001` 改为 `0.01`，另存输出；解释收敛是否更快、是否不稳定。至少三个 seed 后再讨论稳健性。

## 4. 实验二：可训练 VAE

<!-- COLAB:04_vae -->

**目标：** 同时训练编码器均值/方差与解码器；记录重建项、KL和总目标，生成重建图与先验样本。

```bash
python examples/course_labs.py vae --train-size 2048 --val-size 512 --epochs 6 --batch-size 64 --latent-dim 16 --beta 1 --output runs/verified-vae
```

输入 `[0,1]`，编码器 MLP 输出16维 mu/logvar，经重参数化采样，再由解码器输出28×28 logits。重建 BCE 按像素求和、按样本平均，KL 按潜变量求和、按样本平均。`beta=1` 使用标准教学负ELBO目标；`beta=0` 则取消先验约束，不应再称同一个概率目标。

输出 `reconstructions.png` 前16张为验证原图，后16张为使用潜变量均值的确定性重建，按每行8张排列；`samples.png` 为标准高斯先验采样经解码后的16张图。验证损失用固定随机数的潜变量采样，重建图则使用均值，所以二者不是同一重建统计。

**验收：** 梯度到达 mu/logvar 层；KL 有限且非负；总损失下降；能区分重建与先验采样。模糊图像是小型MLP和目标函数的实际局限，不应通过只展示最好的样本掩盖。

**对照练习：** 固定数据与 seed，对比 beta 为0.1/1/4。分别报告重建项和KL，不能只比较加权后总损失；同一固定先验噪声比较样本变化。

## 5. 实验三：DDPM 完整训练与反向采样

<!-- COLAB:05_ddpm -->

**目标：** 训练带时间嵌入的小型 U-Net 噪声预测器，从纯噪声迭代生成图片，而非停留在正向加噪。

```bash
python examples/course_labs.py ddpm --train-size 2048 --val-size 512 --epochs 10 --batch-size 64 --steps 200 --output runs/verified-ddpm
```

输入归一化到 `[-1,1]`；随机抽时间步，用闭式公式加噪，优化噪声 MSE。网络含正弦时间编码、残差卷积、GroupNorm和跳跃连接。每轮验证固定时间/噪声，使曲线可比；`zero_predictor_mse` 是直接预测零噪声的基线，正常约为1。

采样采用固定后验方差和裁剪的 x0 估计，从 `steps-1` 迭代到0；最后一步不加噪。`samples.png` 为固定种子16张生成图；`forward_noising.png` 的4行分别为早期、中早、中晚、最后一步，每行16张同一验证图像，相同 eps 便于观察信号如何被淹没。显示时做范围裁剪，不表示训练噪声分布被截断。

**验收：** 预测误差低于零预测基线；生成图中能看到服饰轮廓；检查是否存在重复、破碎或模糊。只有MSE下降而无图像检查，不能完成验收。本实验未实现FID评估，不能报告FID或与大型论文模型排位比较。

**对照练习：** 比较100/200步，分别从头训练匹配设置；记录终端累计alpha、训练预算和生成时间。若仅改变已训练模型的时间索引并跳步，那不是本采样公式的有效消融。

## 6. 实验四：DINOv2 冻结特征线性评价

<!-- COLAB:06_probe -->

**目标：** 判断预训练表征是否能在少量标签上支持分类，比较相同ViT架构的随机初始化骨干。

```bash
python examples/course_labs.py probe --train-size 1024 --val-size 256 --epochs 20 --batch-size 16 --lr 0.01 --random-baseline --output runs/verified-probe
```

首次从官方 GitHub 下载代码，从 Meta 官方服务器下载约84MiB权重；`torch.hub` 会执行所加载仓库的代码，本项目固定官方提交 `e1277af2ba9496fbadf7aec6eba56e8d882d1e35`，避免随 main 漂移。可用 `--hub-dir` 指向已下载的官方本地仓库；离线还需预先缓存权重。只替换成 tiny 不算完成本实验。

预处理把灰度复制为3通道、缩放224×224、使用ImageNet均值/标准差。官方ViT-S/14输出384维特征。骨干 `eval()` 且 `requires_grad=False`；先提取特征，再用训练特征均值和标准差标准化，训练单层线性头。随机骨干对照使用相同架构、划分、预处理和训练轮数；分类头重置同一随机种子。此处没有使用数据增广或论文原版全基准训练配方。

输出包括两组各自的 `*_history.json`、最佳 `*_probe.pt` 与汇总 `metrics.json`。检查点保存线性头、标准化统计和官方编码器标识；重新推断还需要相同骨干与权重，未把整个预训练模型重复打包。当前 probe 不支持断点续训，需重新提特征与训练头；CNN/VAE/DDPM支持恢复。

**验收：** 预训练和随机骨干都成功评价；骨干参数没有梯度；比较绝对差值而非只报预训练成绩。若差距不明显，先检查预处理、加载与样本量，再照实分析数据集过于简单等原因。

## 7. 恢复、最终测试与复现

例如先训练CNN一轮，再续到总共三轮，必须在同一输出目录保留最佳检查点：

```bash
python examples/course_labs.py cnn --train-size 2048 --val-size 512 --epochs 1 --output runs/resume-demo
python examples/course_labs.py cnn --train-size 2048 --val-size 512 --epochs 3 --output runs/resume-demo --resume runs/resume-demo/last.pt
```

恢复校验模式、划分、seed、batch、学习率、潜变量等关键设置，并恢复优化器和随机数状态。`--epochs` 是目标总轮数，不是再追加的轮数。跨设备/库版本仍可能出现数值差异，本项目不保证GPU逐位重复。

设置锁定后才可为CNN或probe加 `--evaluate-test`，运行到指定轮数并按验证集选取模型，然后评价完整官方测试集。本次三种子对照报告验证分数，另对seed7的CNN设置执行一次最终测试，见[实测记录](evidence.html)。DINO未报告最终测试分数。若读者反复用测试结果选参数，需另留独立评估数据。

## 8. 排错与资源预算

| 症状 | 先检查 | 可执行处理 |
|---|---|---|
| 找不到脚本或依赖 | 当前目录、虚拟环境、解释器版本 | 回到仓库根目录；`python -c "import torch; print(torch.__version__)"`。 |
| 数据/权重下载失败 | 首次网络、代理、作者源是否可达 | 重试并保留报错；数据目录可迁移；DINO官方本地仓库用hub-dir。 |
| CUDA显存不足 | batch、224分辨率、其他GPU程序 | probe先改batch为4；或device cpu、缩小样本做流程检查。 |
| DINO出现Python类型注解错误 | 是否改用main或其他提交 | 使用默认固定兼容提交；新提交建议Python3.10以上。 |
| 损失NaN | 数据范围、lr、VAE logvar、非法mask | 用单个batch打印范围和有限性；恢复默认学习率。 |
| VAE重建尚可但采样差 | KL权重、潜变量分布、训练不足 | 检查KL和先验采样；记录beta对照，避免只看重建。 |
| DDPM输出雪花 | 数据归一化、末端alpha、反向索引 | 对照数学章和默认200步设置，不以8步冒烟结果评价质量。 |
| 曲线波动或指标不一致 | 样本数、seed、拆分、最佳与最后模型 | 比较配置/索引哈希；用best做评价；汇总多种子。 |

具体耗时依硬件、网络与数据缓存而变。首轮下载时间不等于训练时间；实测记录明确报告环境和含下载的墙钟耗时。原始数据、预训练权重与训练检查点不进入发布ZIP，来源与重新获取命令完整保留。
