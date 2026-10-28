# 从数学到视觉模型：推导、例题与实现

本章与[知识总览](tutorial.html)、[四项完整实验](practice.html)、[练习答案](answers.html)配套。每节包含定义、关键推导、数值例子和检查任务。先在纸上算一次，再用张量验证；外部课程用于扩展，不作为省略本章解释的理由。

也可进入[在线代码实验室](playground.html)使用全部数学示例，或打开PyTorch / TensorFlow教学Notebook。本页代码面板在浏览器本机执行，每次运行变量相互独立，首次需要下载Python与数学库；停止会重置环境。

## 1. 张量、线性代数与形状

图像批次通常记为 $X\in\mathbb R^{B\times C\times H\times W}$。线性层使用 $Y=XW^T+b$，其中输入最后一维为 $d_{in}$，权重形状为 $d_{out}\times d_{in}$。矩阵乘法保留前导批次维，收缩匹配的内维；逐元素乘法则依赖广播。

例如 $X=[1,2]$，$W=[3,4]$，$b=1$，输出 $12$。不是 $[4,9]$，后者是另一种逐元素运算。广播从尾维对齐，维度相等或其中一个为 1 才合法。对 `[B,N,D]` 添加 `[D]` 合法；添加 `[N]` 往往不合法，或在碰巧同维时产生语义错误。

内积衡量对齐程度，范数衡量大小。单位化向量的内积就是余弦相似度。协方差描述各特征共同变化，中心化数据的协方差为 $X^TX/(n-1)$。特征向量给出方差方向；SVD 将矩阵分解为 $U\Sigma V^T$，截取最大的奇异值可得到低秩近似。这里的“低秩”是保留主要线性变化方向，不能保证保留所有类别信息。

**检查任务：** 写出 `[8,196,384] @ [384,64]` 的结果形状，并解释为什么不能把空间维随意当成通道维。延伸：[D2L 线性代数](https://d2l.ai/chapter_preliminaries/linear-algebra.html)。

<!-- RUNNER:matrix -->

## 2. 导数、链式法则与自动微分

设 $y=wx+b$，$L=(y-t)^2/2$。先对损失求局部导数，再沿计算图相乘：

$$\frac{\partial L}{\partial w}=(y-t)x,\qquad \frac{\partial L}{\partial b}=y-t.$$

取 $x=2,w=3,b=1,t=5$，有 $y=7,L=2,\partial L/\partial w=4,\partial L/\partial b=2$。梯度下降以 $w\leftarrow w-\eta\nabla_w L$ 更新参数。PyTorch 的反向传播累加 `.grad`，所以每个优化步都需要 `optimizer.zero_grad(set_to_none=True)`。

数值梯度检查使用中心差分 $[L(w+h)-L(w-h)]/(2h)$。用 float64 和适中的 $h$，才能减少舍入与截断误差；它用于检查实现，不适合替代大模型的反向传播。`detach()` 切断图，`no_grad()` 禁止记录，二者均不能凭空让参数更新。延伸：[自动微分](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

<!-- RUNNER:gradient -->

## 3. 概率、似然、熵与交叉熵

条件概率 $p(a\mid b)=p(a,b)/p(b)$，贝叶斯公式把似然与先验转成后验。独立样本的联合似然是乘积，取对数后成为求和；最小化负对数似然等价于最大化似然。

分类 logits 为 $z$，概率为 $p_k=\exp(z_k)/\sum_j\exp(z_j)$，真实类为 $y$ 时 $L=-\log p_y$。稳定实现先减去最大 logit，不改变 softmax。取 logits `[2,0]`、标签 0，有 $L=\log(1+e^{-2})\approx0.126928$。其梯度为 $\partial L/\partial z_k=p_k-\mathbf1[k=y]$。

`CrossEntropyLoss` 接收 logits 和整数类别，不能先 softmax 再交给它。二分类的 `BCEWithLogitsLoss` 同样接收 logits，内部稳定合并 sigmoid 与对数。熵 $H(p)=-\sum p\log p$ 表示分布的不确定性；KL 散度 $D_{KL}(p\Vert q)=\mathbb E_p\log(p/q)$ 非负且不对称。延伸：[D2L 概率](https://d2l.ai/chapter_preliminaries/probability.html)。

## 4. VAE：为什么要引入 ELBO

生成模型假设 $p_\theta(x,z)=p(z)p_\theta(x\mid z)$。真实后验通常难算，引入可计算的编码分布 $q_\phi(z\mid x)$。在分布支撑适当、相关期望存在的条件下：

$$\log p_\theta(x)=\log\mathbb E_{q_\phi}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\geq\mathbb E_{q_\phi}\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}=\mathcal L(x).$$

不等式来自 log 的凹性与 Jensen 不等式。展开联合概率得到：

$$\mathcal L(x)=\mathbb E_q\log p_\theta(x\mid z)-D_{KL}(q_\phi(z\mid x)\Vert p(z)).$$

还可以直接得到精确差距：

$$\log p_\theta(x)-\mathcal L(x)=D_{KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))\geq0.$$

因此训练最大化下界，而不是直接算出真实似然。本项目最小化负下界：重建负对数似然加 KL。额外乘 $\beta\ne1$ 是一种权衡目标，一般不再等于原模型的标准负 ELBO。

### 对角高斯 KL 的推导

设 $q=\mathcal N(\mu,\operatorname{diag}(\sigma^2))$，$p=\mathcal N(0,I)$。将高斯对数密度相减，利用 $\mathbb E_q(z_i-\mu_i)^2=\sigma_i^2$、$\mathbb E_qz_i^2=\mu_i^2+\sigma_i^2$，得到：

$$D_{KL}(q\Vert p)=\frac12\sum_i(\mu_i^2+\sigma_i^2-1-\log\sigma_i^2).$$

实现保存 `logvar`，即 $\log\sigma^2$，所以 `exp(logvar)` 是方差，`exp(0.5*logvar)` 才是标准差。例：一维 $\mu=1,\sigma=2$，KL 为 $[1+4-1-\log4]/2\approx1.306853$；$\mu=0,\sigma=1$ 时为 0。

### 重参数化与重建项

用 $z=\mu+\sigma\epsilon,\epsilon\sim\mathcal N(0,I)$ 把随机性移到与参数无关的噪声，编码器可以通过 $z$ 获得路径导数。把 `z` detach 会破坏这一梯度路径。

本实验输出 Bernoulli logits，逐像素 BCE 求和后再对样本求平均；灰度像素处于 `[0,1]`，该做法是常用的教学重建目标，对连续灰度并非严格的离散 Bernoulli 观测生成假设。换成固定方差高斯时，重建项与平方误差成比例。BCE、MSE、逐像素均值与逐图求和的数值不能直接比较；改 reduction 后必须重新解释 KL 的相对权重。

**检查任务：** 分别画出重建项、KL 项、总损失，确认不是靠关闭 KL 或输出常数得到“下降”。阅读：[VAE 原论文](https://arxiv.org/abs/1312.6114)、[Stanford生成模型课程：VAE](https://deepgenerativemodels.github.io/notes/vae/)。

<!-- RUNNER:gaussian-kl -->

## 5. DDPM：从逐步加噪到闭式采样

设 $\beta_t\in(0,1)$，$\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^t\alpha_s$。正向过程：

$$q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I).$$

把 $x_{t-1}$ 的表达式代入，独立高斯噪声的方差相加，有 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$。归纳得到：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).$$

这使训练时能直接随机抽取一个时间步，无需真的做前面的全部加噪。例：$\bar\alpha_t=0.25,x_0=2,\epsilon=-1$，则 $x_t=1-\sqrt{0.75}\approx0.133975$。这里的单个标量只用于验算公式，实际图像会先归一化到 `[-1,1]`。

### 条件后验与反向均值

令待求变量为 $u=x_{t-1}$。由Bayes公式，固定 $x_t,x_0$ 后，后验正比于 $q(x_t\mid u)q(u\mid x_0)$。忽略与u无关的常数，其对数为：

$$-\frac{\Vert x_t-\sqrt{\alpha_t}u\Vert^2}{2\beta_t}-\frac{\Vert u-\sqrt{\bar\alpha_{t-1}}x_0\Vert^2}{2(1-\bar\alpha_{t-1})}=-\frac12 A\Vert u\Vert^2+B^Tu+\mathrm{const}.$$

展开平方，精度系数与一次项分别为：

$$A=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})},\qquad B=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0.$$

配方为 $-A\Vert u-B/A\Vert^2/2+\mathrm{const}$，所以方差是 $A^{-1}$、均值是 $B/A$。整理便得到下面两式。这个推导先用于t>1；t=1时以零方差的退化后验单独处理：

$$\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,$$

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.$$

模型不能看到生成时未知的 $x_0$，因此用噪声预测 $\epsilon_\theta(x_t,t)$ 估计：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.$$

不做裁剪时，代入后等价于 $\mu_\theta=\alpha_t^{-1/2}[x_t-\beta_t\epsilon_\theta/\sqrt{1-\bar\alpha_t}]$。本项目将 $\hat x_0$ 裁剪到 `[-1,1]` 再使用后验系数；这是稳定小实验的选择，不应与未裁剪的式子混为完全相同的算法。

### 训练与采样是两条流程

训练抽 $x_0,t,\epsilon$，构造 $x_t$，最小化 $\mathbb E\Vert\epsilon-\epsilon_\theta(x_t,t)\Vert^2$；简化的均匀噪声 MSE 不是把变分下界的所有时间权重原样保留。采样从 $x_T\sim\mathcal N(0,I)$ 开始，倒序循环均值和后验噪声，最后一步不再加噪。

代码索引 `0` 对应数学 $t=1$，此时前一项 $\bar\alpha_0=1$，后验方差为 0。默认 200 步线性 beta 从 `1e-4` 到 `0.05`，末端累计 alpha 接近 0，才适合从近似纯噪声开始；冒烟测试的 8 或 16 步只检查程序流程，不能验证生成质量。应查看生成图，噪声 MSE 下降不能独立证明样本好看或覆盖充分。

### DDIM 与无分类器引导

DDIM 可在所选的递减时间子序列上构造更新；$\eta=0$ 的更新在固定初始噪声下是确定性的，但它不是简单跳过 DDPM 循环步骤。无分类器引导的一种约定为 $\hat\epsilon=\epsilon_u+s(\epsilon_c-\epsilon_u)$，这里 $s=0$ 表示无条件、$s=1$ 表示条件预测。有些文章用另一种偏移约定，比较超参数前先看定义。当前教学代码为无条件 DDPM，没有实现 DDIM 或条件引导，不能把这些扩展当成已验证功能。

阅读：[DDPM](https://arxiv.org/abs/2006.11239)、[DDIM](https://arxiv.org/abs/2010.02502)、[MIT 扩散与流课程](https://diffusion.csail.mit.edu/2025/)。

<!-- RUNNER:diffusion -->

## 6. ODE、SDE 与 Flow Matching

ODE 给定初值与速度场 $dx/dt=v(x,t)$，Euler 更新 $x_{k+1}=x_k+h v(x_k,t_k)$。对 $v=-x,x_0=1,h=0.1$，一步是 0.9，精确值为 $e^{-0.1}\approx0.904837$；缩小步长用于检查离散误差。

SDE 多了 Brownian 随机增量：$dX_t=f(X_t,t)dt+g(t)dW_t$。Euler–Maruyama 更新为 $X_{k+1}=X_k+hf(X_k,t_k)+g(t_k)\sqrt h\,\xi_k$，$\xi_k\sim\mathcal N(0,I)$。噪声系数是 $\sqrt h$ 而非 $h$；因为 Brownian 增量方差为 $h$。当 $f=0,g=1$ 时，很多轨迹在时间 $t$ 的样本方差应接近 $t$。

Flow Matching 的简单条件路径为 $x_t=(1-t)x_0+tx_1$，条件目标速度为 $x_1-x_0$。平方误差最优解是给定当前位置的条件期望速度，学习的是对应边缘分布的速度场。独立配对端点的直线路径不自动等于最优传输路径；采样器步数、耦合和速度参数化都会影响结果。

**练习：** 比较 Euler 步长 0.1 与 0.05 的终点误差；模拟 10000 条 Brownian 路径验证方差。系统教学：[MIT 课程讲义与带解答实验](https://diffusion.csail.mit.edu/2025/)。

<!-- RUNNER:sde -->

## 7. 最优传输与二分匹配

离散 OT 给定质量 $a_i,b_j$ 与代价 $C_{ij}$，求运输计划：

$$\min_{P\geq0}\sum_{ij}P_{ij}C_{ij},\qquad P\mathbf1=a,\quad P^T\mathbf1=b.$$

对 $a=b=(1/2,1/2)$、$C=\begin{bmatrix}1&4\\3&2\end{bmatrix}$，对角运输的总代价为 1.5，交叉为 3.5。若改成“两个任务各分配一次”的指派问题，每条边质量为 1，代价分别是 3 和 7，不能混淆归一化。

熵正则 OT 在代价上加入 $\varepsilon\sum P_{ij}(\log P_{ij}-1)$，令 $K_{ij}=\exp(-C_{ij}/\varepsilon)$，交替缩放 $u=a/(Kv)$、$v=b/(K^Tu)$ 得到 Sinkhorn 迭代。小 epsilon 会引起数值问题，实际常用 log 域实现；本课程不把这段概念式当作大规模稳定求解器。

DETR 的集合预测把预测框与真实框构成代价矩阵，求一对一匹配，再计算匹配后的分类与框损失。匹配是离散分配，不能理解成每个框独立选择最相似目标。矩阵 `[[4,1,3],[2,0,5],[3,2,2]]` 的最优列序为 `[1,0,2]`，总代价 5。文献常称 Hungarian matching；SciPy 当前 `linear_sum_assignment` 文档说明使用改进的 Jonker–Volgenant 算法，名称应区分。

教学：[作者的 Computational Optimal Transport 教材](https://optimaltransport.github.io/book/)、[SciPy 指派问题定义与例子](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)、[DETR 作者实现](https://github.com/facebookresearch/detr)。

<!-- RUNNER:assignment -->

## 8. 注意力与视觉 Transformer

先线性投影 $Q=XW_Q,K=XW_K,V=XW_V$，单头输出为：

$$A=\operatorname{softmax}(QK^T/\sqrt{d_k}+M),\qquad Y=AV.$$

softmax 沿“被关注的 key”一维计算，因此每个 query 的权重和为 1；被禁止的位置在 mask 中加负无穷。缩放项缓解随维度增长的点积方差，但不能代替 LayerNorm。多头各自投影并拼接后还要输出投影。仅写 `Q=K=V=X` 可以演示形状，不能代表完整可训练注意力。

图像划分为 patch，每个 patch 展平并投影为 token，加位置编码，进入“LayerNorm → 注意力 → 残差 → LayerNorm → MLP → 残差”的常见 pre-norm 模块。位置编码使模型知道空间位置，残差提供较直接的梯度路径，LayerNorm 沿特征维归一化。若 224×224 图像使用 16×16 patch，图像 token 数是 196，加 CLS 后是 197。注意力矩阵的内存随 token 数平方增长。

**检查任务：** 运行[参考实现下载](solutions.py)，验证投影权重得到梯度；将一个 key 屏蔽后检查对应权重为 0。阅读：[ViT](https://arxiv.org/abs/2010.11929)、[D2L 注意力章节](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)。

<!-- RUNNER:attention -->

## 9. InfoNCE、DINO 与表征评价

InfoNCE 让 anchor 与正样本的相似度相对负样本更高：

$$L_i=-\log\frac{\exp(s_{i,+}/\tau)}{\sum_{j\in\mathcal C_i}\exp(s_{ij}/\tau)}.$$

候选集合包含正样本，通常排除 anchor 自身；不同实现是否包含其他视图需要明确。若候选只有一个正样本相似度 1、一个负样本相似度 0，$\tau=0.5$，则损失同样为 $\log(1+e^{-2})$。温度改变分布锐度，负样本来源与增广决定任务难度。

DINO 自监督学习用教师分布指导学生分布，不是直接采用上述负样本对比损失。教师输出经过中心化和温度调整；教师不反传，参数以学生的 EMA 更新：$\theta_T\leftarrow m\theta_T+(1-m)\theta_S$。不同视图之间计算交叉熵并配合多裁剪等机制。只用“teacher = student”加 MSE 不等于复现 DINO。DINOv2 还包含更完整的数据和训练目标设计；不要与目标检测论文 DINO 混淆。

表征评价要冻结编码器，只训练线性分类头；特征标准化统计只从训练集计算。比较同架构随机初始化编码器，可以帮助判断提升是否来自预训练表征。当前实验使用官方 DINOv2 ViT-S/14、固定提交与冻结骨干，属于迁移评价；没有从头训练 DINO，也不复现论文的完整大规模基准。

教学：[UvA SimCLR 完整 Notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)、[DINO 作者代码](https://github.com/facebookresearch/dino)、[DINOv2 模型卡](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)。

## 10. 优化、泛化与可信指标

训练集用于更新参数，验证集用于选择超参数和检查点，测试集在设置锁定后评价。反复根据测试集调参会产生信息泄漏，即使没有执行反向传播。生成实验也应保存固定的验证噪声与时间步，让不同轮的 MSE 更可比。

SGD 使用梯度方向，动量积累历史；Adam 对梯度的一阶、二阶矩进行指数平均和偏差修正，AdamW 将权重衰减与自适应梯度更新解耦。学习率决定每步大小，batch size 影响噪声和吞吐，随机种子改善重复性，但不保证所有硬件逐位一致。

样本平均损失必须按实际样本数加权。若两个 batch 大小为 64 和 6，不能简单平均两个 batch 的平均损失。评价时同时用 `model.eval()` 和 `inference_mode()`；前者改变 Dropout/BatchNorm 行为，后者关闭梯度记录，作用不同。

混淆矩阵约定行为真实类、列为预测类。$\mathrm{Precision}=TP/(TP+FP)$，$\mathrm{Recall}=TP/(TP+FN)$。分割有 $\mathrm{IoU}=TP/(TP+FP+FN)$ 与 $\mathrm{Dice}=2TP/(2TP+FP+FN)$。零分母时必须说明约定。例：TP=1,FP=1,FN=1 时 IoU=1/3、Dice=1/2。类别不平衡时只看 accuracy 可能误导；报告每类或宏平均指标。

**结束检查：** 能否不看公式写出 VAE 正负号、DDPM 训练与采样的区别、attention 的归一化方向，以及 DINO 教师为何不接收梯度？用[答案与参考代码](answers.html)核对，再按[考核标准](assessment.html)提交证据。
