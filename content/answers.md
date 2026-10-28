# 练习、参考答案与纠错

建议先完成题目，记录自己的推导和程序输出，再展开核对。答案给出关键步骤；仅复述结论不等于掌握。[数学教学](math.html)解释原理，[完整实验](practice.html)把公式接到训练流程。

## A. 数学与形状：10 题

| 题目 | 参考答案与理由 |
|---|---|
| A1：`[8,196,384] @ [384,64]` 得到什么？ | `[8,196,64]`。收缩 384 维，前导维保留。 |
| A2：224×224 图像、patch16，加入 CLS 后多少 token？ | 196+1=197；patch8 则为 784+1=785。仅比较图像 token 的注意力矩阵，元素数增加 16 倍。 |
| A3：32 像素、核3、padding1、stride2、dilation1 的卷积输出？ | `floor((32+2-2-1)/2+1)=16`。通道数由卷积核数量决定。 |
| A4：$y=wx+b,L=(y-t)^2/2$，x=2,w=3,b=1,t=5？ | y=7，L=2，dw=4，db=2。链式法则先求 y 的梯度 2，再乘 x。 |
| A5：logits `[2,0]`、标签0 的交叉熵？ | `log(1+exp(-2))=0.126928`。输入 CrossEntropy 的是 logits。 |
| A6：一维高斯均值1、标准差2，相对标准高斯的 KL？ | `0.5*(1+4-1-log(4))=1.306853`。不要把标准差 2 代入方差位置。 |
| A7：`logvar=log(4)`，如何采样？ | `z=mu+exp(0.5*logvar)*eps=mu+2*eps`。用 `exp(logvar)` 会把标准差错误地变成4。 |
| A8：两个 batch 的平均损失分别2和5，大小64和6？ | `(2*64+5*6)/70=2.257143`；简单平均为3.5，是错误的样本平均值。 |
| A9：TP=1,FP=1,FN=1，IoU 与 Dice？ | IoU=1/3，Dice=1/2。分母体现两个指标对重叠的不同计数。 |
| A10：单位向量 `[1,0]` 与 `[0,1]` 的余弦相似度？ | 0，表示正交；并非概率为0，也不表示两个图像必然语义不同。 |

## B. 生成模型与表征：10 题

| 题目 | 参考答案与理由 |
|---|---|
| B1：ELBO 和真实对数似然之间差多少？ | 差 `KL(q(z|x) || p(z|x))`，因此下界不会超过真实对数似然。下界较高并不能单独证明后验近似精确。 |
| B2：最小化 VAE 损失时 KL 前面是什么号？ | 正号。标准目标是重建负对数似然加 KL。写成减 KL 会鼓励后验无限偏离先验。 |
| B3：累计 alpha=0.25，x0=2，eps=-1，xt？ | `sqrt(.25)*2 + sqrt(.75)*(-1)=0.133975`。 |
| B4：DDPM 最后一次反向更新为什么不加噪？ | 数学 t=1 时前一累计 alpha 为1，条件后验方差为0。代码索引为0。 |
| B5：噪声 MSE 从1降到0.08，能证明图像质量高吗？ | 不能。需同时看固定种子采样、覆盖与失败例；MSE只是训练代理指标。 |
| B6：`CFG=eps_u+s*(eps_c-eps_u)`，s=0/1 是什么？ | 0为无条件，1为条件预测。另一种公式约定中的 s 数值不能直接搬来。 |
| B7：SDE 步长变成原来1/4，单步噪声标准差怎么变？ | 变为1/2。Brownian 增量的标准差与步长平方根成比例。 |
| B8：InfoNCE 正相似度1、负相似度0、温度0.5，损失？ | 0.126928；候选中包含正样本，共两个候选。若额外负样本或自身进入分母，答案就不同。 |
| B9：DINO 教师如何更新？ | 教师不反传，通过学生参数的指数移动平均更新；输出还涉及中心化、温度和跨视图目标。单独 EMA 不是完整 DINO。 |
| B10：线性评价时 backbone 可以一起训练吗？ | 不可以。应冻结 backbone，并只训练线性头；若一起训练应称微调并重新说明设置。 |

## C. 算法与评价：6 题

| 题目 | 参考答案与理由 |
|---|---|
| C1：x'=-x，初值1，Euler 步长0.1，一步？ | 0.9；精确值 `exp(-.1)=.904837`。差异是数值离散误差。 |
| C2：OT 两边质量均为(.5,.5)，代价`[[1,4],[3,2]]`？ | 对角运输总代价1.5；若是一人一任务指派，总代价3。注意质量规范。 |
| C3：代价`[[4,1,3],[2,0,5],[3,2,2]]` 最优指派？ | 行0→列1，行1→列0，行2→列2，总代价1+2+2=5。逐行贪心不保证全局合法最优。 |
| C4：attention softmax 沿哪一维？ | key维；每个 query 的可见 key 权重和为1。完全屏蔽一行会产生未定义 softmax，需要阻止。 |
| C5：标准化均值/方差能从 train+val 一起估计吗？ | 不能；这把验证信息引入训练。仅用训练特征估计，再应用到验证与测试。 |
| C6：训练准确率很高、验证下降时怎么办？ | 检查划分和泄漏；保留最佳验证检查点；尝试增广、正则化或降低容量；不能通过不断查看测试集选方案。 |

## D. 四项编程练习与参考实现

下载[练习骨架](exercises.py)，逐个实现；原理见[数学章](math.html)，可运行参考代码见 [solutions.py](solutions.py) 和 [course_labs.py](course_labs.py)。在仓库根目录执行 `python examples/solutions.py`。

### D1：对角高斯 KL

```python
def diagonal_gaussian_kl(mu, logvar):
    return (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)).mean()
```

检查两个情形：全零 mu/logvar 得到0；mu=1、logvar=log4 得到1.306853。额外与 `torch.distributions.kl_divergence` 比较，避免只测试零输入。

### D2：闭式加噪

```python
def add_noise(x0, alpha_bar, epsilon):
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * epsilon
```

对 `[B,C,H,W]`，先将选出的累计 alpha 变成 `[B,1,1,1]`。固定 x0，大量独立 eps 的均值应接近信号项，方差应接近 `1-alpha_bar`。

### D3：带 mask 的注意力

```python
def masked_attention(q, k, v, allowed):
    if not allowed.any(-1).all():
        raise ValueError("each query needs a visible key")
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    return weights @ v, weights
```

检查屏蔽位置权重为0，行和为1。`solutions.py` 再加上训练所需的 Q/K/V 和输出投影；用输出平方均值反传，查看投影参数的非零梯度。

### D4：按样本数汇总

```python
def sample_weighted_mean(batch_means, batch_sizes):
    return sum(m*n for m, n in zip(batch_means, batch_sizes)) / sum(batch_sizes)
```

要求长度相同且总样本数大于0；生产代码还应检查非法输入。测试必须包含不能整除 batch size 的数据集。仅用等长 batch 会漏掉这个错误。

## E. 常见错误定位顺序

先检查数据范围和 shape，再检查单个 batch 的前向与反向，接着尝试过拟合极小训练集，最后扩大数据。损失 NaN 时查看输入、logvar、学习率和梯度；VAE 模糊需同时检查模型容量、KL 权重与观测模型；DDPM 像雪花需检查归一化、时间索引、末端 alpha 和反向均值。

DINO 验证接近随机猜测时先检查灰度复制成3通道、224分辨率、ImageNet归一化、`eval()` 与预训练权重实际加载。官方 backbone 的 xFormers 缺失警告在本课程固定版本中可回退普通注意力；完整 DINO 训练的依赖要求不同。

无法下载资料应保留错误和源链接，不要用合成数据的结果替代真实数据指标。冒烟数据上得到高准确率仅说明这套合成任务容易，不能证明表征泛化。
