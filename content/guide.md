# 视觉论文阅读清单与路线

整理日期：2026-10-28。精选30篇，提供公开全文来源。

范围：扩散/流匹配、视觉Transformer、VAE及视觉编码、自监督DINO和相邻视觉基础模型。以经典方法和代表性演进为主，并非完整综述或2026年最新论文排名。标题、作者和首发日期主要通过arXiv官方API核对；beta-VAE依据作者主页。阅读年份按会议年份或标明的预印本年份排列，可能不同于arXiv首发年份。

## 推荐阅读顺序

1. **架构基础**：Transformer → ViT → Swin。原始Transformer论文研究机器翻译，作为视觉方法的前置基础。
2. **生成主线**：VAE → VQ-VAE / VQGAN → DDPM → DDIM → CFG → Latent Diffusion → DiT。之后阅读Score-SDE、Consistency Models与Flow Matching。
3. **视觉表征主线**：SimCLR / MoCo → DINO → iBOT → DINOv2 → Registers → DINOv3；与MAE对照理解不同自监督目标。
4. **视觉任务与多模态**：DETR → 目标检测DINO；CLIP → SAM，用来理解图文对齐与提示分割。

## 概念区分

- 自监督DINO（Caron等）与目标检测DINO（Zhang等）是不同工作。
- MAE的autoencoder不等同于变分自编码器VAE；VQGAN也包含对抗训练与离散视觉编码。
- Flow Matching是相邻的连续流生成方法，不应直接当成DDPM的同义词。
- 只下载论文PDF，不另行获取补充附件；公开PDF内原本包含的附录予以保留。arXiv公开稿可能与最终出版版不同。

## 分组清单

### 扩散与流匹配

| 编号 | 简称 / 论文 | 年份与会议 | 优先级 | 阅读重点 |
|---|---|---|---|---|
| 01 | **DDPM**：[ Denoising Diffusion Probabilistic Models ](https://arxiv.org/abs/2006.11239) | 2020 · NeurIPS 2020 | 必读 | 扩散生成基础：前向加噪、反向去噪与噪声预测训练目标。 |
| 02 | **DDIM**：[ Denoising Diffusion Implicit Models ](https://arxiv.org/abs/2010.02502) | 2021 · ICLR 2021 | 必读 | 理解非马尔可夫采样，以及少步生成与采样轨迹。 |
| 03 | **Score-SDE**：[ Score-Based Generative Modeling through Stochastic Differential Equations ](https://arxiv.org/abs/2011.13456) | 2021 · ICLR 2021 | 进阶 | 用随机微分方程统一得分模型与扩散模型，连接概率流ODE。 |
| 04 | **CFG**：[ Classifier-Free Diffusion Guidance ](https://arxiv.org/abs/2207.12598) | 2022 · arXiv / NeurIPS 2021 Workshop | 必读 | 无分类器引导：条件与无条件预测结合，控制条件遵循程度。 |
| 05 | **Latent-Diffusion**：[ High-Resolution Image Synthesis with Latent Diffusion Models ](https://arxiv.org/abs/2112.10752) | 2022 · CVPR 2022 | 必读 | 将扩散放入压缩潜空间；理解自编码器、交叉注意力与文本条件生成。 |
| 06 | **DiT**：[ Scalable Diffusion Models with Transformers ](https://arxiv.org/abs/2212.09748) | 2023 · ICCV 2023 | 必读 | 用Transformer替代U-Net，连接视觉Transformer与潜空间扩散。 |
| 07 | **Consistency-Models**：[ Consistency Models ](https://arxiv.org/abs/2303.01469) | 2023 · ICML 2023 | 进阶 | 通过一致性映射实现一步或少步生成；可与扩散蒸馏对照。 |
| 08 | **Flow-Matching**：[ Flow Matching for Generative Modeling ](https://arxiv.org/abs/2210.02747) | 2023 · ICLR 2023 | 进阶 | 学习连续流的速度场；属于相邻生成路线，并不等同于DDPM。 |

### 视觉Transformer

| 编号 | 简称 / 论文 | 年份与会议 | 优先级 | 阅读重点 |
|---|---|---|---|---|
| 09 | **Transformer**：[ Attention Is All You Need ](https://arxiv.org/abs/1706.03762) | 2017 · NeurIPS 2017 | 必读 | 自注意力与Transformer原始论文；任务是机器翻译，作为视觉Transformer前置基础。 |
| 10 | **ViT**：[ An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ](https://arxiv.org/abs/2010.11929) | 2021 · ICLR 2021 | 必读 | 图像切成patch token，再用Transformer进行图像分类。 |
| 11 | **DeiT**：[ Training data-efficient image transformers & distillation through attention ](https://arxiv.org/abs/2012.12877) | 2021 · ICML 2021 | 拓展 | 数据高效训练与蒸馏token，理解ViT训练策略。 |
| 12 | **Swin**：[ Swin Transformer: Hierarchical Vision Transformer using Shifted Windows ](https://arxiv.org/abs/2103.14030) | 2021 · ICCV 2021 | 必读 | 分层表示与移位窗口注意力，面向检测和分割的通用视觉骨干。 |
| 13 | **DETR**：[ End-to-End Object Detection with Transformers ](https://arxiv.org/abs/2005.12872) | 2020 · ECCV 2020 | 必读 | 集合预测、object query与二分图匹配，建立端到端目标检测框架。 |
| 14 | **MAE**：[ Masked Autoencoders Are Scalable Vision Learners ](https://arxiv.org/abs/2111.06377) | 2022 · CVPR 2022 | 必读 | 高比例掩码与像素重建，理解视觉自监督预训练；MAE不是VAE。 |

### VAE与视觉编码

| 编号 | 简称 / 论文 | 年份与会议 | 优先级 | 阅读重点 |
|---|---|---|---|---|
| 15 | **VAE**：[ Auto-Encoding Variational Bayes ](https://arxiv.org/abs/1312.6114) | 2014 · ICLR 2014 | 必读 | ELBO、重参数化技巧和连续潜变量生成模型的基础。 |
| 16 | **beta-VAE**：[ beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework ](https://www.matthey.me/publication/beta_vae/) | 2017 · ICLR 2017 | 必读 | 调整KL项权重探索解耦表征；不能保证在任意数据上自动得到真实生成因素。 |
| 17 | **VQ-VAE**：[ Neural Discrete Representation Learning ](https://arxiv.org/abs/1711.00937) | 2017 · NeurIPS 2017 | 必读 | 向量量化和离散码本，将图像转成可供生成模型使用的离散潜变量。 |
| 18 | **VQ-VAE-2**：[ Generating Diverse High-Fidelity Images with VQ-VAE-2 ](https://arxiv.org/abs/1906.00446) | 2019 · NeurIPS 2019 | 拓展 | 分层离散潜表示与高保真图像生成，接续VQ-VAE阅读。 |
| 19 | **NVAE**：[ NVAE: A Deep Hierarchical Variational Autoencoder ](https://arxiv.org/abs/2007.03898) | 2020 · NeurIPS 2020 | 进阶 | 深层分层VAE及其训练和架构设计，研究高分辨率图像建模。 |
| 20 | **VQGAN**：[ Taming Transformers for High-Resolution Image Synthesis ](https://arxiv.org/abs/2012.09841) | 2021 · CVPR 2021 | 必读 | 融合感知压缩、对抗训练与Transformer先验；属于离散视觉编码与生成路线。 |

### 自监督DINO

| 编号 | 简称 / 论文 | 年份与会议 | 优先级 | 阅读重点 |
|---|---|---|---|---|
| 21 | **DINO-SSL**：[ Emerging Properties in Self-Supervised Vision Transformers ](https://arxiv.org/abs/2104.14294) | 2021 · ICCV 2021 | 必读 | 无标签自蒸馏、teacher-student和多裁剪训练，观察ViT的语义特征。 |
| 22 | **DINOv2**：[ DINOv2: Learning Robust Visual Features without Supervision ](https://arxiv.org/abs/2304.07193) | 2023 · arXiv 2023 / TMLR 2024 | 必读 | 规模化数据整理和自监督训练，获得可迁移的视觉通用特征。 |
| 23 | **Registers**：[ Vision Transformers Need Registers ](https://arxiv.org/abs/2309.16588) | 2024 · ICLR 2024 | 进阶 | 用register token改善ViT特征图中的伪影，关联DINOv2的稠密特征。 |
| 24 | **DINOv3**：[ DINOv3 ](https://arxiv.org/abs/2508.10104) | 2025 · arXiv 2025 | 进阶 | 研究更大规模自监督视觉特征与稠密表示；与DINO、DINOv2形成演进线。 |
| 25 | **iBOT**：[ iBOT: Image BERT Pre-Training with Online Tokenizer ](https://arxiv.org/abs/2111.07832) | 2022 · ICLR 2022 | 进阶 | 在线视觉tokenizer与掩码预测，连接DINO式自蒸馏和局部patch表征。 |

### 其他视觉基础

| 编号 | 简称 / 论文 | 年份与会议 | 优先级 | 阅读重点 |
|---|---|---|---|---|
| 26 | **CLIP**：[ Learning Transferable Visual Models From Natural Language Supervision ](https://arxiv.org/abs/2103.00020) | 2021 · ICML 2021 | 必读 | 图像与文本对比学习及零样本迁移，连接视觉表征和语言条件生成。 |
| 27 | **SAM**：[ Segment Anything ](https://arxiv.org/abs/2304.02643) | 2023 · ICCV 2023 | 必读 | 提示驱动的通用分割基础模型，关注点、框和掩码提示。 |
| 28 | **SimCLR**：[ A Simple Framework for Contrastive Learning of Visual Representations ](https://arxiv.org/abs/2002.05709) | 2020 · ICML 2020 | 拓展 | 对比学习中的数据增强、投影头与批量大小，作为自监督基础对照。 |
| 29 | **MoCo**：[ Momentum Contrast for Unsupervised Visual Representation Learning ](https://arxiv.org/abs/1911.05722) | 2020 · CVPR 2020 | 拓展 | 动量编码器与队列字典，理解视觉对比学习如何构造负样本。 |
| 30 | **DINO-Detection**：[ DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection ](https://arxiv.org/abs/2203.03605) | 2023 · ICLR 2023 | 拓展 | 改进去噪anchor query的DETR检测器；与Meta的自监督DINO系列不同。 |

## 获取论文全文

在[论文目录](index.html)中点击公开 PDF 来源，或在项目根目录运行 `python scripts/download_papers.py`。下载的第三方论文遵循各自许可证。
