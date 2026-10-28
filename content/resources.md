# 知识点与可靠教学来源

总览中保留了76个学习入口。本页补足容易缺失的进阶知识，并指定学习内容、练习和完成标准；先读本站[数学章](math.html)，再按下表到官方课程或作者教材深学。外部页面可能调整，课程年份和章节名称比网页行号更稳定。

| 知识点 | 教学来源与应学内容 | 本站练习与完成标准 |
|---|---|---|
| 线性代数、导数、概率 | [D2L 数学预备](https://d2l.ai/chapter_preliminaries/index.html)：线性代数、微积分、自动微分、概率。 | math第1–3节；完成A1–A8，梯度数值验证。 |
| PyTorch全训练流程 | [官方 Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：Tensor、Dataset、transform、模型、autograd、优化与保存加载。 | 实验CNN；能解释每一步并恢复检查点。 |
| CNN与视觉任务 | [Stanford CS231n课程](https://cs231n.stanford.edu/2025/schedule.html)：卷积、优化、识别/检测/分割、自监督。 | 总览视觉章节、混淆矩阵与IoU例题；不把分类指标套给生成。 |
| VAE与变分推断 | [Stanford生成模型课程：VAE](https://deepgenerativemodels.github.io/notes/vae/)配合[VAE作者论文](https://arxiv.org/abs/1312.6114)：后验难点、下界、重参数化。 | math第4节，B1–B2，VAE完整实验；正确区分方差和标准差。 |
| attention/Transformer | [D2L 注意力与Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)：QKV、多头、位置、编码器。 | math第8节，D3与投影梯度检查；了解ViT patch映射。 |
| DDPM与score | [MIT Diffusion/Flow Matching 2025](https://diffusion.csail.mit.edu/2025/)：课程讲义“Diffusion Models”及lab代码/解答；配合[DDPM](https://arxiv.org/abs/2006.11239)。 | math第5节，B3–B6；完整反向采样和噪声基线。 |
| ODE/SDE与数值积分 | [同一MIT课程](https://diffusion.csail.mit.edu/2025/)的ODE/SDE基础讲义和Lab 1：速度、Brownian运动、数值模拟。 | math第6节；Euler步长误差与Brownian方差实验。普通ODE教程不能单独覆盖SDE。 |
| Flow Matching | [MIT课程](https://diffusion.csail.mit.edu/2025/)的conditional/marginal paths及Flow Matching讲义、Lab 2。 | 解释条件速度的回归目标，区分独立耦合与OT耦合。 |
| 最优传输与Sinkhorn | [Peyré/Cuturi作者教材与资源](https://optimaltransport.github.io/book/)：离散运输、熵正则、Sinkhorn；书内从离散问题逐步阅读。 | math第7节与C2；检查质量约束与代价尺度。 |
| 一对一匹配 | [SciPy官方定义、示例](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)与[DETR作者代码](https://github.com/facebookresearch/detr)：指派问题、匹配代价和匹配后损失。 | math第7节与C3；手算最优指派，并解释与逐框贪心差异。 |
| InfoNCE/SimCLR | [UvA完整教学Notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)：多视图、正负样本、温度、实现、线性评价。 | math第9节与B8；写清分母候选集合、检查自身mask。 |
| DINO/DINOv2 | [DINO作者代码](https://github.com/facebookresearch/dino)与[DINOv2模型卡](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)：EMA教师、中心化、温度、预训练骨干与评价范围。 | B9–B10；四实验中的真实预训练与随机骨干线性评价。 |
| 可信评价与数据泄漏 | [scikit-learn官方常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html)：预处理一致性、数据泄漏与随机性；配合PyTorch训练流程。 | C5–C6；仅用train估计特征统计，保留test，记录多种子。 |

这些来源承担不同角色：课程/教材用于系统学习，原论文用于核对方法定义，API文档用于核对具体软件行为。仅浏览链接不构成掌握；每个主题应至少留下一个手算结果和一个可运行检查。本站尚未提供大型检测、分割、Flow Matching全模型训练，这些属于进一步项目，不能把阅读链接当作已经完成实现。
