# 视觉论文前置知识完整教程

**适用目标**：读懂本资料夹中的 30 篇扩散模型、视觉 Transformer、VAE、DINO 与相关视觉论文，并能够独立实现小规模实验。  
**学习方式**：按“理解概念 → 手算一个例子 → 写最小代码 → 阅读对应论文 → 复现实验”的顺序。  
**时间参考**：有 Python 基础者约 12–16 周、每周 8–12 小时；时间是估算，不是能力保证。  
**配套资料**：[论文阅读指南](guide.md)｜[可搜索论文目录](index.html)｜[可运行 PyTorch 实验](vision_basics_lab.py)。

> 本教程覆盖读这些论文的核心前置知识与入门实践，不替代完整的数学或深度学习教材。每节给出“合格标准”；达到标准后就可以进入对应论文，不必等所有知识都学完。

## 0. 开始前：准备、术语与学习验收

### 0.1 环境

建议掌握 Python 函数、类、列表/字典、文件路径、虚拟环境、NumPy 数组。先在终端检查：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

安装命令应按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 的操作系统和硬件选项生成；CPU 也能完成本教程的小实验。无需大型数据集即可运行 `python vision_basics_lab.py`。若要训练真实分类模型，先从 FashionMNIST 或 CIFAR-10 开始。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| Python 语法、函数、类 | [Python 官方教程（完整课程）](https://docs.python.org/3/tutorial/) | 先学第 3–9 章，再写函数和类。 |
| NumPy 数组、索引、形状 | [NumPy Quickstart（官方）](https://numpy.org/doc/stable/user/quickstart.html) | 边读边做数组形状与索引练习。 |
| PyTorch 安装和设备 | [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/) | 按本机系统和 CPU/CUDA 选择命令。 |

### 0.2 四层学习标准

| 层级 | 你能做什么 | 证据 |
|---|---|---|
| 识别 | 说出术语和任务 | 能解释输入、输出、损失和数据 |
| 推导 | 从公式推出张量形状和目标 | 能手算二维或小批量例子 |
| 实现 | 写出能训练、验证的最小模型 | 代码运行，损失下降，指标可复现 |
| 研究 | 解释论文选择并做公平对照 | 有基线、消融、误差分析和实验记录 |


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 从概念到实验的完整路径 | [斯坦福 CS231n 课程与作业](https://cs231n.stanford.edu/2025/) | 用课程安排和作业检验能否实现与分析。 |

### 0.3 共通约定

图像张量在 PyTorch 中常用 `[B,C,H,W]`：批量、通道、高、宽。ViT 中常用 `[B,N,D]`：批量、token 数、通道维。务必在每次 `reshape/permute/flatten` 前写出形状。`train/validation/test` 三份数据各司其职：训练更新参数，验证选择超参数，测试只用于最后报告。所有指标都要注明数据集、划分、预处理和评价协议。

---


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 图像和 token 张量形状 | [PyTorch 张量教程](https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) | 重点练习 shape、reshape、索引和设备。 |
| 数据划分与避免泄漏 | [scikit-learn 常见错误：数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html) | 重点看训练/测试划分和预处理拟合顺序。 |

## 1. 数学基础：按论文需要学习

### 1.1 线性代数与张量

**必会内容**：向量、矩阵、转置、矩阵乘法、内积、范数、正交、特征值/奇异值的直觉；张量的形状、广播、批量矩阵乘法。矩阵乘法要求内维相等：`[B,N,D] @ [B,D,M] -> [B,N,M]`。广播要从右侧维度对齐，长度必须相同或其中一个为 1。

**视觉例子**：224×224 RGB 图像，patch 大小 16×16，共有 `(224/16)^2=196` 个 patch。每个 patch 展平后有 `16×16×3=768` 个值。线性投影到 `D=768` 后，ViT 输入形状为 `[B,196,768]`；加入一个分类 token 后为 `[B,197,768]`。

**自注意力形状推导**：`Q=XW_Q`、`K=XW_K`、`V=XW_V`。若 `X∈R^{B×N×D}`，单头维度为 `d`，则 `Q,K,V∈R^{B×N×d}`，`QKᵀ∈R^{B×N×N}`，`softmax(QKᵀ/√d)V∈R^{B×N×d}`。`N×N` 注意力矩阵解释了标准全局注意力随 token 数增长的内存压力。多头注意力把通道拆成若干头，输出拼接后再投影。

**练习**：用纸算 `2×3` 矩阵乘 `3×2` 矩阵；在 PyTorch 中分别打印 `x.shape`、`x.transpose(-2,-1).shape` 和 `x @ x.transpose(-2,-1)` 的形状。

**合格标准**：不靠运行代码，也能说出卷积特征图、patch tokens、Q/K/V 与注意力权重的形状。对应 Transformer、ViT、Swin、DETR、DiT、DINO。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 向量、矩阵、内积、范数 | [《动手学深度学习》2.3 线性代数（中文）](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html) | 含代码与练习，先完成本节。 |
| 矩阵分解、特征值、SVD | [MIT 18.06 线性代数（完整课程）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | 按需学习特征值与奇异值章节。 |
| 广播规则 | [NumPy Broadcasting（官方）](https://numpy.org/doc/stable/user/basics.broadcasting.html) | 手算维度对齐再运行例子。 |
| Q/K/V 与多头形状 | [《动手学深度学习》多头注意力（中文）](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html) | 对照本文的 `[B,N,D]` 形状推导。 |

### 1.2 微积分、链式法则与反向传播

函数 `y=f(x)` 的导数表示局部变化率；多变量函数用偏导/梯度。链式法则：若 `L=g(f(x))`，则 `dL/dx=(dL/df)(df/dx)`。神经网络反向传播就是把这个规则沿计算图重复应用。

**手算例子**：`y=wx+b`，`L=(y-t)^2/2`，所以 `∂L/∂w=(y-t)x`、`∂L/∂b=y-t`。一次 SGD 更新是 `w←w−η∂L/∂w`。梯度不是直接“修改参数”；优化器读取梯度后才更新。多步累积梯度前通常要清零。

**常见坑**：在训练前错误调用 `detach()` 会切断梯度；原地操作可能破坏反向传播需要的值；把 `model.eval()` 当成关闭梯度是误解——它主要切换 Dropout/BatchNorm 行为，评估还需 `torch.inference_mode()` 或 `torch.no_grad()`。

**练习**：手算 `x=2,w=3,b=1,t=5` 的 `L`、`∂L/∂w`、`∂L/∂b`；再用 PyTorch `backward()` 对照。

**合格标准**：能解释 `loss.backward()`、`optimizer.step()`、`zero_grad()` 的顺序，并能定位“梯度为 None”的问题。对应全部 30 篇。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 导数、偏导、梯度、链式法则 | [《动手学深度学习》2.4 微积分（中文）](https://zh.d2l.ai/chapter_preliminaries/calculus.html) | 完成链式法则与梯度练习。 |
| 多元微积分系统学习 | [MIT 18.02SC 多元微积分（完整课程）](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) | 重点看偏导、梯度与链式法则单元。 |
| 自动微分和计算图 | [PyTorch 自动微分教程（官方）](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html) | 运行例子并对照手算梯度。 |

### 1.3 概率、统计与信息论

**概率必会**：随机变量、条件概率、贝叶斯公式 `p(z|x)=p(x|z)p(z)/p(x)`、期望、方差、协方差、独立与条件独立；正态分布 `N(μ,σ²)`；联合分布与边缘化 `p(x)=∫p(x,z)dz`。模型里的 `p(x|z)` 是“给定潜变量生成数据”的条件分布，`q(z|x)` 常是近似后验。

**统计必会**：样本均值、标准误、置信区间直觉、随机种子与重复运行、分布偏移。训练集指标高不证明泛化；评价必须区分随机波动和真实改进。分类还要理解类别不平衡与混淆矩阵。

**信息论必会**：熵 `H(p)=−E_p log p`，交叉熵 `H(p,q)=−E_p log q`，KL 散度 `D_KL(q||p)=E_q log(q/p)≥0`。交叉熵可用于分类；KL 在 VAE 中约束近似后验与先验，在蒸馏和分布匹配中也会出现。KL 不对称：`D_KL(p||q)` 与 `D_KL(q||p)` 通常不同。

**练习**：对二分类 `p=(1,0)`、预测 `q=(0.8,0.2)` 计算交叉熵 `−log 0.8`；把预测改成 `(0.2,0.8)` 再比较。对一维高斯 `q=N(μ,σ²)` 与 `p=N(0,1)`，记住 `KL=0.5(μ²+σ²−1−log σ²)`，检查 `μ=0,σ=1` 时为 0。

**合格标准**：读 VAE 的 ELBO、DINO/CLIP 的分布或对比目标时，能说清概率分布、目标和优化方向。对应 VAE、beta-VAE、NVAE、DINO、CLIP、扩散模型。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 条件概率、贝叶斯、期望方差 | [《动手学深度学习》2.6 概率（中文）](https://zh.d2l.ai/chapter_preliminaries/probability.html) | 先学概念、再做小题。 |
| 概率论系统课程 | [MIT 6.041SC 概率论（完整课程与习题）](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) | 需要更完整的概率训练时按课程顺序学。 |
| 交叉熵与分类概率 | [《动手学深度学习》Softmax 回归（中文）](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html) | 理解 logits、softmax 与交叉熵。 |
| KL 与变分推断 | [斯坦福 CS236：VAE 教学笔记](https://deepgenerativemodels.github.io/notes/vae/) | 结合 ELBO 逐项理解 KL 的作用。 |

### 1.4 优化、数值稳定与泛化

经验风险最小化：`min_θ (1/n)Σ_i ℓ(f_θ(x_i),y_i)`。SGD 使用小批量梯度近似全数据梯度；Adam/AdamW 使用动量与自适应尺度。学习率通常比“换一个新模块”更影响能否稳定训练。权重衰减、数据增强、早停、学习率调度控制过拟合或训练动态。

**数值知识**：`logsumexp`/稳定 softmax，梯度裁剪，混合精度，过大/过小学习率，NaN/Inf 排查。`softmax` 之前直接 `exp` 很大数会溢出；分类任务优先给 `CrossEntropyLoss` 原始 logits，避免重复 softmax。训练日志至少记录 loss、验证指标、学习率、随机种子、耗时。

**练习**：同一个小模型比较学习率 `1e-4`、`1e-3`、`1e-1`；画训练/验证曲线，描述欠拟合、过拟合和发散。

**合格标准**：能判断性能变化来自模型还是训练设置，能排查不收敛。对应 DeiT、DINOv2/v3、DiT、NVAE 等训练敏感论文。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| SGD 与小批量优化 | [《动手学深度学习》随机梯度下降（中文）](https://zh.d2l.ai/chapter_optimization/sgd.html) | 手算一次参数更新。 |
| Adam 和优化器 | [《动手学深度学习》Adam（中文）](https://zh.d2l.ai/chapter_optimization/adam.html) | 与 SGD 对比收敛曲线。 |
| 训练循环与学习率 | [PyTorch 优化教程（官方）](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | 重点看 zero_grad/backward/step。 |
| 数据泄漏与泛化 | [scikit-learn 常见错误（官方）](https://scikit-learn.org/stable/common_pitfalls.html) | 检查先划分数据再拟合预处理。 |

### 1.5 生成模型专项数学

**VAE 的 ELBO**：

`log p_θ(x) ≥ E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x)||p(z))`。

前一项鼓励重建/生成质量，后一项约束潜变量分布。常见重参数化：`z=μ_φ(x)+σ_φ(x)⊙ε`，`ε∼N(0,I)`；随机性移到不依赖参数的 `ε`，梯度能传到 `μ,σ`。beta-VAE 改成 `重建项−β·KL`，更大 β 常加强约束，但不保证真实因素自动解耦。离散 VQ-VAE 采用码本量化，不是把上述连续高斯公式直接套上。

**DDPM 的前向加噪**：给定 `α_t=1−β_t`、`ᾱ_t=∏_{s≤t}α_s`，可一次采样 `x_t=√ᾱ_t x_0+√(1−ᾱ_t) ε`，`ε∼N(0,I)`。最小训练目标常写成 `E||ε−ε_θ(x_t,t)||²`。训练是预测噪声，采样是从噪声逐步回到图像；两者不要混淆。DDIM 改变采样路径，CFG 组合有条件/无条件预测以调节条件引导。

**Flow Matching 的直觉**：学习随时间变化的速度场 `v_θ(x,t)`，让样本沿连续流从简单分布到数据分布；它与扩散模型有关联，但目标和采样表述需要单独读。Score-SDE 则用连续时间随机微分方程表述得分模型，要求对微分方程有基本直觉。

**练习**：画出 `ᾱ_t` 从接近 1 降到接近 0 时 `x_t` 的信噪变化；手算 `μ=0,σ=1` 的 VAE KL；运行配套脚本中的 `concepts_demo()`。

**合格标准**：能逐项解释 ELBO 与 DDPM 噪声预测公式，并清楚区分 VAE、VQ-VAE、扩散、Flow Matching。对应 01–08、15–20。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| VAE、ELBO、重参数化 | [斯坦福 CS236 深度生成模型：VAE](https://deepgenerativemodels.github.io/notes/vae/) | 从潜变量到 ELBO 按顺序阅读。 |
| 生成模型完整课程 | [斯坦福 CS236 开放讲义目录](https://deepgenerativemodels.github.io/notes/) | 补生成模型、变分推断的整体框架。 |
| DDPM 前向/反向过程 | [Hugging Face 扩散课程 Unit 1](https://huggingface.co/learn/diffusion-course/en/unit1/1) | 跟随 notebook 做加噪和去噪实验。 |
| 潜空间扩散与文本条件 | [Hugging Face 扩散课程 Unit 3](https://huggingface.co/learn/diffusion-course/en/unit3/1) | 观察编码器、U-Net、文本条件与解码器。 |

### 1.6 进阶数学何时补

| 主题 | 什么时候需要 | 要掌握到什么程度 |
|---|---|---|
| 常微分/随机微分方程 | Score-SDE、Flow Matching、Consistency Models | 状态随时间变化、漂移/噪声项、数值积分直觉 |
| 最优传输 | 深读 Flow Matching | 耦合、路径、速度场的概念即可起步 |
| 互信息与对比估计 | CLIP、SimCLR、MoCo | 正负样本、温度、InfoNCE 目标 |
| 集合匹配/匈牙利算法 | DETR、检测 DINO | 一对一匹配为什么避免重复预测 |
| 频域与图像信号处理 | 扩散细节、图像伪影分析 | 低/高频、卷积与采样伪影直觉 |

---


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| ODE 与数值积分 | [MIT 18.03 微分方程（完整课程）](https://ocw.mit.edu/courses/18-03-differential-equations-spring-2010/) | 学一阶微分方程和欧拉数值法。 |
| 生成模型的连续时间背景 | [斯坦福 CS236 讲义目录](https://deepgenerativemodels.github.io/notes/) | 为 Score-SDE 与 Flow Matching 补先修。 |
| 注意力与对比目标的基础 | [《动手学深度学习》注意力机制（中文）](https://zh.d2l.ai/chapter_attention-mechanisms/index.html) | 先掌握相似度、softmax 和 Q/K/V。 |
| 目标检测和框匹配前置 | [《动手学深度学习》目标检测与边界框（中文）](https://zh.d2l.ai/chapter_computer-vision/bounding-box.html) | 先能计算 IoU 再读 DETR。 |

## 2. PyTorch：从张量到可靠实验

### 2.1 张量操作与设备

掌握 `torch.tensor/zeros/randn`、`dtype`、`device`、`shape`、索引、切片、`view/reshape`、`permute`、`flatten`、`unsqueeze`、广播、`matmul/einsum`。`reshape` 可能复制数据；`view` 对内存连续性有限制。图像网络输入通常是 float32 `[B,C,H,W]`，分类标签通常是 int64 `[B]`。把模型和输入放在同一设备：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(8, 3, 224, 224).to(device)
patches = torch.nn.functional.unfold(x, kernel_size=16, stride=16)
print(patches.shape)  # [8, 768, 196]
```

**练习**：将上面的结果转成 `[8,196,768]`；逆向解释每个数字来源。**验收**：能独立查出 shape mismatch 出现在哪一层。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 张量、索引、形状、设备 | [PyTorch 张量教程（官方）](https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) | 完整做完操作示例。 |
| NumPy 与广播对照 | [NumPy Quickstart（官方）](https://numpy.org/doc/stable/user/quickstart.html) | 对照数组的 dtype、shape 和广播。 |
| 矩阵乘法与展开 | [PyTorch torch.matmul 文档](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) | 查批量维的行为，再写小例子。 |

### 2.2 自动微分、模块与损失

定义网络时继承 `nn.Module`，在 `__init__` 注册层，在 `forward` 定义计算。`nn.Parameter` 会被模块管理。`requires_grad` 决定是否跟踪梯度；冻结预训练骨干时可对参数调用 `requires_grad_(False)`。`loss.backward()` 计算梯度，`optimizer.step()` 更新参数。

常用层：`Conv2d`、`Linear`、`LayerNorm`、`BatchNorm2d`、`ReLU/GELU`、`Dropout`、`Embedding`、`MultiheadAttention`。常用损失：`CrossEntropyLoss`（多类分类、输入 logits）、`BCEWithLogitsLoss`（多标签/二类独立输出）、`MSELoss`（回归/重建/噪声预测）。辨别“二分类两个 logits 的交叉熵”和“单 logit 的 BCE”实现差别。

**练习**：写 `nn.Linear(10,3)`，输入 `[4,10]`，检查输出 `[4,3]`；冻结该层后查看梯度。**验收**：能说明每个参数的形状和梯度是否存在。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| nn.Module 与网络层 | [PyTorch 构建神经网络教程](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) | 自己改层数和输出维度。 |
| 自动微分与冻结参数 | [PyTorch Autograd 机制](https://docs.pytorch.org/docs/stable/notes/autograd.html) | 分清 requires_grad、no_grad、eval。 |
| 分类/重建损失 | [PyTorch 优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | 从 logits 到 loss 再反传。 |

### 2.3 Dataset、DataLoader 与数据增强

`Dataset.__len__` 给样本数，`Dataset.__getitem__` 返回单个样本及标签；`DataLoader` 做批处理、打乱和多进程读取。训练集可做随机裁剪、翻转、颜色扰动；验证/测试集使用确定性的预处理。检测与分割的图像和框/掩码必须一起变换，不能只变换图像。新项目可参考 torchvision 的 transforms v2 文档。

**数据泄漏例子**：先对全数据集做均值/方差拟合再切分训练/测试，会把测试信息带入训练。正确顺序是先划分，再只用训练集计算统计量。按患者/视频/设备等相关实体划分，避免同源帧同时进入训练和测试。

**练习**：给配套脚本的 `make_data` 添加随机亮度变化，只在训练集使用；验证集保持原样。**验收**：能打印一个 batch 的图像、标签形状，并可视化增强前后。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| Dataset 与 DataLoader | [PyTorch 数据集与加载器教程](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) | 写自己的 __getitem__ 和 batch。 |
| 图像增强与标准化 | [《动手学深度学习》图像增广（中文）](https://zh.d2l.ai/chapter_computer-vision/image-augmentation.html) | 辨别训练与验证增强。 |
| 框、掩码同步增强 | [TorchVision transforms v2 入门](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html) | 重点看图像、BoundingBoxes、Mask 同步变换。 |
| 防止数据泄漏 | [scikit-learn 常见错误（官方）](https://scikit-learn.org/stable/common_pitfalls.html) | 只在训练集上拟合统计量。 |

### 2.4 完整训练循环与评估

最小顺序：`model.train()` → 遍历 batch → `zero_grad` → 前向 → 损失 → `backward` → `step`；验证时 `model.eval()` + `torch.inference_mode()`。每轮记录训练损失和验证指标，最好保存最佳验证权重。不要在训练集上挑模型后声称是测试性能。

配套的 [`vision_basics_lab.py`](vision_basics_lab.py) 给出可直接运行的 CNN 分类示范，数据是随机生成的“亮方块在左/右”图像，避免外部下载。它还演示注意力形状、VAE 重参数化、扩散前向加噪、`state_dict` 保存和恢复。运行：

```bash
python vision_basics_lab.py
```

**练习**：修改 batch size、学习率和训练轮数，观察验证正确率；将 `model.eval()` 暂时去掉，加入 Dropout 后观察差别。**验收**：运行三次可解释结果波动；恢复的模型对同一输入得到相同输出。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 训练、验证、优化器 | [PyTorch Quickstart（官方）](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) | 先运行端到端范例。 |
| 训练循环逐步拆解 | [PyTorch 优化教程（官方）](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | 按步骤实现 train/eval 两个函数。 |
| 本地无下载练习 | [配套 PyTorch 小实验](vision_basics_lab.py) | 先运行，再改学习率和模型结构。 |

### 2.5 检查点、复现与性能

保存 `model.state_dict()`；恢复时构建同样结构并 `load_state_dict(torch.load(path, weights_only=True, map_location=device))`，推理再调用 `eval()`。恢复训练通常还需保存优化器状态、轮次与调度器状态。实验记录固定种子、数据划分、代码版本、超参数和硬件；固定随机种子并不保证所有硬件操作完全确定。

显存不足时依次检查 batch size、分辨率、token 数、是否在验证阶段保留计算图；可考虑梯度累积或混合精度。对于 `float16` 混合精度，官方示例使用 `torch.autocast` 与 `torch.amp.GradScaler`。初学者先让 float32 训练正确，再优化性能。

**练习**：对同一模型保存“只含权重”的检查点和“可继续训练”的检查点，列出两者内容差别。**验收**：从检查点恢复后能完成预测；继续训练的 loss 曲线接续原实验。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| state_dict 与继续训练 | [PyTorch 模型保存/加载（官方）](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | 区分推理权重与完整训练检查点。 |
| 混合精度 | [PyTorch AMP 示例（官方）](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) | 先读典型训练例子，再决定是否使用。 |
| 迁移学习与冻结骨干 | [PyTorch 视觉迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | 分别尝试冻结与微调。 |

### 2.6 最小实现关卡

| 关卡 | 实现内容 | 检查点 |
|---|---|
| P1 | NumPy/torch 实现线性回归 | 手算梯度与 autograd 相符 |
| P2 | CNN 分类 | 正确区分训练/验证、保存最佳模型 |
| P3 | 单头注意力和 patchify | 权重每行求和为 1，形状正确 |
| P4 | 简化 VAE | 重建、KL 和总损失均有记录 |
| P5 | DDPM 前向加噪 | `t` 增大时图像逐渐接近噪声 |
| P6 | 用现成骨干提取特征 | 冻结与微调的评价协议相同 |

---


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| P1–P3：基础实现 | [PyTorch 基础课程（官方完整路径）](https://docs.pytorch.org/tutorials/beginner/basics/) | 逐章完成张量、数据、网络、训练、保存。 |
| P4–P5：生成模型动手 | [Hugging Face 扩散模型课程](https://huggingface.co/learn/diffusion-course/en/unit0/1) | 先做 Unit 1，再读 VAE 与扩散论文。 |
| P6：预训练特征迁移 | [PyTorch 迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | 用相同数据划分对比冻结与微调。 |

## 3. 计算机视觉：任务、模型与评价

### 3.1 数字图像与预处理

图像是采样后的像素阵列。区分像素坐标、通道顺序 RGB/BGR、uint8 `0–255` 与 float `0–1`、均值方差归一化、插值、缩放、裁剪和长宽比。卷积核聚合局部邻域，步幅改变采样间隔，padding 影响边缘和输出尺寸；下采样可能损失细节。图像分类可接受某些几何变化，医学/遥感等领域增强必须符合任务语义。

卷积输出高度公式：`H_out=floor((H+2P−D(K−1)−1)/S+1)`，宽度同理；其中 `K` 核大小、`S` 步幅、`P` padding、`D` 膨胀。**练习**：计算输入 32、核 3、padding 1、stride 2 的输出尺寸 16。**验收**：能独立解释输入图像预处理与模型预训练设置不一致为何导致性能下降。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 图像、通道、色彩与数组 | [Pillow 图像概念（官方）](https://pillow.readthedocs.io/en/stable/handbook/concepts.html) | 观察像素、通道顺序和裁剪。 |
| 卷积核、填充、步幅 | [《动手学深度学习》卷积与填充（中文）](https://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html) | 手算输出尺寸后用代码验证。 |
| 视觉预处理与增广 | [《动手学深度学习》图像增广（中文）](https://zh.d2l.ai/chapter_computer-vision/image-augmentation.html) | 重点比较训练和验证处理。 |

### 3.2 分类、检测、分割、检索与生成

| 任务 | 输入 → 输出 | 常见指标 | 对应论文 |
|---|---|---|---|
| 分类 | 图像 → 类别 | accuracy、macro-F1、top-k | ViT、DeiT、Swin |
| 目标检测 | 图像 → 类别与边框集合 | IoU、AP/mAP | DETR、检测 DINO |
| 语义分割 | 图像 → 每像素类别 | mIoU、Dice | SAM 可作提示分割基础 |
| 实例分割 | 图像 → 各实例掩码 | mask AP | SAM |
| 图像检索/表征 | 图像 → 嵌入向量 | Recall@K、线性探测 | DINO、DINOv2/v3、CLIP |
| 图像生成 | 噪声/条件 → 图像 | 人评、FID 等 | VAE、扩散、DiT |

指标不能跨任务互换。准确率在严重类别不平衡时会误导；检测 AP 依赖 IoU 阈值和评价协议；生成指标需说明真实样本数量、特征提取器和预处理。视觉模型结果也需要查看定性样例与失败样例。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 分类与准确率 | [《动手学深度学习》图像分类数据集（中文）](https://zh.d2l.ai/chapter_linear-networks/image-classification-dataset.html) | 从单标签分类开始。 |
| 检测与边界框 | [《动手学深度学习》目标检测和边界框（中文）](https://zh.d2l.ai/chapter_computer-vision/bounding-box.html) | 先练标注、坐标和 IoU。 |
| 语义/实例分割 | [《动手学深度学习》语义分割与数据集（中文）](https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html) | 区别每像素类别与实例掩码。 |
| 分类指标选择 | [scikit-learn 模型评价（官方）](https://scikit-learn.org/stable/modules/model_evaluation.html) | 查 accuracy、precision、recall、F1 适用条件。 |

### 3.3 CNN、残差网络与 U-Net

CNN 用局部卷积和权重共享提取特征；随层数增加，感受野变大、分辨率常下降。残差连接 `y=x+F(x)` 帮助深网络优化。U-Net 以编码器提取多尺度特征、解码器恢复空间分辨率、跳连保留细节，是理解扩散去噪网络的重要前置结构。

**练习**：画出 28×28 → 14×14 → 7×7 的特征图大小与通道数；实现一个两层 CNN 分类器。**验收**：能解释为何分割比分类更需要保留空间位置。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| CNN、感受野、卷积 | [《动手学深度学习》卷积神经网络整章（中文）](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html) | 从卷积到 LeNet 完整学习。 |
| 残差连接与 ResNet | [《动手学深度学习》ResNet（中文）](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html) | 画残差块并比较普通堆叠。 |
| U-Net 编解码与跳连 | [U-Net 原论文（作者原始公开稿）](https://arxiv.org/abs/1505.04597) | 重点看结构图，结合 CNN 课程理解。 |

### 3.4 Transformer 在视觉中的用法

原始 Transformer 解决序列建模；ViT 把图像切成 patch，添加位置编码后用自注意力处理。全局注意力便于跨远距离信息交互，但 token 多时计算/显存压力大。Swin 用窗口及移位窗口构建分层特征；DeiT 重视数据高效训练和蒸馏；MAE 掩住大量 patch 并重建被遮挡内容。

**三个问题**：① patch 太大，细节会怎样？② 224×224、patch16 与 patch8 的 token 数分别是多少（196 与 784），注意力矩阵元素数约相差几倍（16 倍）？③ MAE 为什么只将可见 patch 送入编码器有计算优势？

**验收**：能从 `[B,C,H,W]` 推导 ViT 输入 `[B,N,D]`，解释位置编码、分类 token、局部窗口与掩码重建。对应 09–14、DiT、DINO。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 注意力与 Transformer | [《动手学深度学习》Transformer（中文）](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html) | 先弄懂 Q/K/V、位置编码和编码器。 |
| 视觉 Transformer 专题 | [斯坦福 CS231n 2025 课程日程](https://cs231n.stanford.edu/2025/schedule.html) | 查 Attention/Transformers 课程材料与作业。 |
| ViT、Swin、MAE 原文 | [本地 30 篇论文目录](index.html) | 按 09–14 顺序结合图和代码阅读。 |

### 3.5 自监督学习、对比学习与蒸馏

自监督从数据本身构造训练信号，无需人工类别标签；它不等于“没有监督目标”。SimCLR 对同一图像的两种增强视图作为正样本，其他图像视图常作负样本；MoCo 用动量编码器和队列维持字典。CLIP 用图文配对做跨模态对比学习。DINO 系列用 teacher-student 自蒸馏，不依赖传统负样本对比训练；DINO 中多裁剪、teacher 更新、输出中心化/温度等防止塌缩的细节很重要。iBOT 结合掩码 patch 预测与自蒸馏。DINOv2/v3 扩大训练与特征迁移能力；Registers 关注稠密特征的异常 token。

**表示塌缩**：若所有输入都映射成相同向量，就无法区分图像。看 loss 下降不足以证明表征有用，还要线性探测、kNN、检索或下游迁移。**练习**：在同一数据划分下比较随机初始化特征与预训练特征的线性分类器。**验收**：能说明每种方法的正样本、负样本/教师、目标和评价方式。对应 21–26、28–29。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 自监督与对比学习课程 | [斯坦福 CS231n 2025 课程日程](https://cs231n.stanford.edu/2025/schedule.html) | 找 Lecture 12：pretext tasks、contrastive learning。 |
| DINO 训练与评价 | [Meta DINO 官方代码与说明](https://github.com/facebookresearch/dino) | 看 teacher-student、预训练模型和线性评价。 |
| DINOv2 与特征迁移 | [Meta DINOv2 官方代码与说明](https://github.com/facebookresearch/dinov2) | 理解冻结特征与下游任务。 |
| 方法演进原文 | [本地 30 篇论文目录](index.html) | 按 SimCLR→MoCo→DINO→iBOT→DINOv2/v3 阅读。 |

### 3.6 生成视觉模型的结构关系

普通 autoencoder 压缩并重建；VAE 在连续概率潜空间中建模；VQ-VAE/VQGAN 用离散码本或量化后的潜变量；Latent Diffusion 把扩散过程放到学习到的压缩空间，降低计算成本；DiT 用 Transformer 处理扩散中的潜空间 patch。不要把“用 autoencoder”直接等同“VAE”。

**条件生成**：模型可以接受类别、文本或图像条件。交叉注意力让图像/潜变量 token 查询文本表示；CFG 在推理时组合有条件与无条件预测。生成结果要同时看条件遵循、质量、多样性和可能的训练数据记忆。**练习**：画出“文本 → 文本编码器 → 交叉注意力 → 潜空间去噪 → 解码成图像”的数据流。**验收**：能解释像素空间和潜空间生成的计算差异。对应 01–08、15–20。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| VAE、潜空间与 ELBO | [斯坦福 CS236 VAE 教学笔记](https://deepgenerativemodels.github.io/notes/vae/) | 先辨别连续潜变量与普通自编码器。 |
| 像素扩散到潜空间扩散 | [Hugging Face 扩散课程 Unit 3](https://huggingface.co/learn/diffusion-course/en/unit3/1) | 跟着图理解编码、去噪、解码。 |
| 完整生成模型课程 | [斯坦福 CS236 开放讲义](https://deepgenerativemodels.github.io/notes/) | 系统补潜变量、生成与评价。 |
| VQ 与扩散原文 | [本地 30 篇论文目录](index.html) | 对照 15–20 与 01–08 的建模空间。 |

### 3.7 检测、分割与多模态

目标检测需表示框坐标、类别、无目标类与重叠程度。IoU=`交集面积/并集面积`。DETR 将预测视为集合，通过一对一匹配对应真实目标；检测 DINO 基于 DETR 路线，和自监督 DINO 不是同一系列。分割需区分语义类别与实例身份；SAM 接受点、框或掩码提示，输出分割掩码。CLIP 的图文共享嵌入支持零样本类别文本匹配，但其相似度不是未经校准的概率。

**练习**：手算两个 4×4 方框水平错开 2 像素时的 IoU（交集 8、并集 24，IoU=1/3）；手动画一个图像里两个同类物体的语义掩码与实例掩码。**验收**：能区分分类、检测、语义分割、实例分割、提示分割的标注和指标。


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 边界框、IoU | [《动手学深度学习》目标检测与边界框（中文）](https://zh.d2l.ai/chapter_computer-vision/bounding-box.html) | 手算 IoU 再写代码。 |
| 检测/实例分割实际训练 | [TorchVision 官方检测微调教程](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) | 看数据格式、框/掩码和评价流程。 |
| 语义分割 | [《动手学深度学习》语义分割（中文）](https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html) | 明确像素类别和实例身份差别。 |
| DETR、检测 DINO、CLIP、SAM | [本地 30 篇论文目录](index.html) | 按 13→30、26→27 顺序读原文。 |

### 3.8 研究实验基本功

论文复现先固定数据集和评估协议，再实现基线。一次只改变一个主要因素；报告参数量、计算预算、训练步数和推理成本。需要做训练/验证曲线、消融、失败案例、不同随机种子。若复制论文中大规模结果需要大量算力，可复现核心机制而非声称达到原始性能。

**最小研究报告模板**：问题与假设；数据和划分；模型及参数量；训练设置；评价指标；主结果表；消融表；错误分析；局限；可复现命令。**验收**：另一人按你的记录能复现大致结论。

---


**逐项学习链接**（推荐按表内顺序学习；“官方”指原项目/课程提供者，论文原文用于进阶对照）：

| 知识点 | 直达教学资料 | 学习目标 |
|---|---|---|
| 实验设计与常见陷阱 | [scikit-learn 常见错误（官方）](https://scikit-learn.org/stable/common_pitfalls.html) | 对照数据泄漏、不一致预处理和随机性。 |
| 评价指标与报告 | [scikit-learn 模型评价（官方）](https://scikit-learn.org/stable/modules/model_evaluation.html) | 先定义指标，再做模型比较。 |
| 完整课程项目训练 | [斯坦福 CS231n 课程与作业](https://cs231n.stanford.edu/2025/) | 参照作业和项目要求写可复现报告。 |

## 4. 30 篇论文的知识依赖地图

| 学习模块 | 先掌握 | 建议读的论文 | 要回答的核心问题 |
|---|---|---|---|
| A 基础架构 | 线性代数、反向传播、CNN、注意力 | Transformer → ViT → DeiT → Swin | 图像如何变成 token？全局与窗口注意力如何取舍？ |
| B 表征学习 | 概率/交叉熵、数据增强、分类评价 | SimCLR → MoCo → DINO → iBOT → DINOv2 → Registers → DINOv3 | 无标签信号怎样产生可迁移特征？ |
| C 自编码与潜变量 | 条件概率、KL、ELBO | VAE → beta-VAE → VQ-VAE → VQ-VAE-2 → NVAE → VQGAN | 连续/离散潜变量各解决什么问题？ |
| D 扩散与流 | 高斯、噪声预测、U-Net、ODE 直觉 | DDPM → DDIM → CFG → Score-SDE → Latent Diffusion → DiT → Consistency Models → Flow Matching | 训练目标与采样路径如何对应？ |
| E 下游任务 | IoU、检测/分割、图文配对 | DETR → 检测 DINO；CLIP → SAM；MAE 可与 DINO 对照 | 表征如何迁移到检测、分割和多模态？ |

注意：上表为**学习顺序**，不是按论文发表时间排序。可按兴趣选生成或表征一条主线深读，另一条先读概览。完整题名、论文来源与已下载 PDF 在[论文目录](index.html)。

---

## 5. 12–16 周执行计划与交付物

| 周次 | 学习内容 | 必交成果 |
|---|---|---|
| 1–2 | Python/NumPy、线性代数、微积分、概率基础 | 形状练习、手算梯度与 KL、简短笔记 |
| 3–4 | PyTorch 张量、autograd、Dataset、训练循环；CNN | 跑通配套脚本，自己训练一个真实小数据集分类器 |
| 5–6 | 图像预处理、分类评价、Transformer/ViT | 手写 patchify 和单头注意力；读 Transformer、ViT、Swin |
| 7–8 | VAE、ELBO、VQ 与潜空间 | 最小 VAE，记录重建和 KL；读 VAE、VQ-VAE、VQGAN |
| 9–10 | DDPM、DDIM、LDM、DiT | 画加噪过程，训练小型噪声预测器；读核心 4 篇 |
| 11–12 | 自监督、DINO、MAE、CLIP | 比较两种特征的线性探测；读 DINO、MAE、CLIP |
| 13–14（可选） | DETR、SAM、检测 DINO；Score-SDE/Flow Matching | 一个检测/分割案例与一份方法对照表 |
| 15–16（可选） | 复现与研究能力 | 一份 4–6 页实验报告，含基线、消融、失败分析 |

每篇论文使用一张读书卡：**任务和输入/输出 → 核心假设 → 模型结构 → 损失 → 训练数据/成本 → 评价协议 → 最关键表格/消融 → 自己的疑问 → 最小复现想法**。优先读摘要、图 1、方法、主结果和消融；遇到推导再回补数学。不要把“读过 PDF”当成“会使用该方法”。

## 6. 最终自测清单

### 数学

- [ ] 能在纸上推导 patch 数、Q/K/V 和注意力矩阵的形状。
- [ ] 能手算单层线性模型的梯度，并用 autograd 验证。
- [ ] 能解释交叉熵、KL、ELBO 和重参数化。
- [ ] 能解释 `x_t=√ᾱ_t x_0+√(1−ᾱ_t)ε` 的每一项。
- [ ] 知道何时需要 ODE/SDE、匹配、互信息等进阶知识。

### PyTorch

- [ ] 能写 `Dataset`、`DataLoader`、`nn.Module`、训练和验证循环。
- [ ] 能处理 `[B,C,H,W]` 与 `[B,N,D]` 的形状变化和设备切换。
- [ ] 能正确使用 `train/eval`、`inference_mode`、`zero_grad/backward/step`。
- [ ] 能保存/加载 `state_dict`，记录随机种子与超参数。
- [ ] 能排查 NaN、显存不足、数据泄漏和训练/验证指标不一致。

### 计算机视觉与论文

- [ ] 能区分分类、检测、语义/实例/提示分割、检索和生成。
- [ ] 能手算 IoU，理解 accuracy、F1、AP/mAP、mIoU、Recall@K 的适用范围。
- [ ] 能说明 CNN、U-Net、ViT、Swin 的信息流和计算差异。
- [ ] 能区分 VAE、MAE、VQ-VAE、VQGAN 和潜空间扩散。
- [ ] 能区分自监督 DINO 与目标检测 DINO。
- [ ] 能为一篇论文设计可复现的最小实验、基线和消融。

**完成标准**：上述大部分问题能脱离资料清楚回答，独立运行一个真实数据集实验，写出结果与局限。达到此程度就有进入视觉算法研究/项目实践的基础；深度仍取决于你选择的细分方向和实验经验。

---

## 7. 可靠的后续学习入口

以下为官方资料，代码细节以你实际安装版本的文档为准：

1. [PyTorch 官方 Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)：Dataset/DataLoader、模型、训练和保存的总览。
2. [PyTorch 优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：训练循环、优化器和损失。
3. [PyTorch Autograd 机制](https://docs.pytorch.org/docs/stable/notes/autograd.html)：梯度、`no_grad`、`inference_mode` 与 `eval` 的区别。
4. [PyTorch 模型保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：`state_dict`、检查点和跨设备加载。
5. [PyTorch 混合精度示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)：`torch.autocast` 和 `torch.amp.GradScaler`。
6. [TorchVision 数据增强文档](https://docs.pytorch.org/vision/stable/transforms.html)：图像以及检测/分割标注的变换。
7. 本资料夹的[30 篇论文清单](guide.md)：每篇的原始论文入口及主题说明。
