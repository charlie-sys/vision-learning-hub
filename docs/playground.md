# 在线代码实验室

边学边改代码：数学小实验在当前网页运行；PyTorch、TensorFlow和视觉模型进入自己的Colab运行环境。无需为本网站创建账号。建议先读[数学教学](math.html)，再做[完整视觉实验](practice.html)。

## 1. 选择运行方式

| 想做什么 | 从哪里开始 | 运行环境 |
|---|---|---|
| 矩阵、梯度、概率、公式验算 | 下方数学代码面板，修改后点击“运行代码” | 当前浏览器CPU，支持兼容的NumPy/SciPy/SymPy/Matplotlib。 |
| PyTorch或TensorFlow张量、自动微分 | 对应基础Notebook的“在Colab运行” | 你的Colab Python环境，需要登录Google账号。 |
| CNN、VAE、DDPM、DINOv2实验 | 对应完整Notebook | Colab CPU或可用GPU；资源与时长以Colab当前规则为准。 |

数学运行器在点击后才下载固定版本Pyodide和数学库，第一次可能需要较长时间；普通阅读不加载Python。代码默认在你的浏览器执行，本站没有接收代码的计算后端。每次运行的变量相互独立，后面的面板不会继承前面变量；依赖会在本页复用缓存。

运行时可点“停止”，会终止计算并重置环境；再次运行需重新初始化。计算阶段超过30秒会停止，环境与库加载阶段上限2分钟。输出最多约64KB，图像最多4张。适合小规模教学，不适合大型训练或长时间任务。修改后的代码可点“下载代码”保存，刷新页面会恢复原始示例。

如果运行环境下载失败，请检查网络后重试，或使用Colab入口。离线直接打开HTML可阅读教程，但运行器需要HTTPS或本地HTTP服务：在仓库根目录执行 `python -m http.server 8000 --directory docs`，访问 `http://localhost:8000/playground.html`。不支持交互式 `input()`，请直接修改代码中的参数。

## 2. 数学实验：矩阵与梯度

矩阵例题的第一项输出应为12。修改权重、输入形状，观察广播与矩阵乘法的区别。有限差分例题应接近手算梯度4。

<!-- RUNNER:matrix -->

<!-- RUNNER:gradient -->

## 3. 概率与注意力

高斯样本的均值/方差只近似理论值；可增加采样量比较误差。注意力归一化沿key维，屏蔽位置的权重应为0。

<!-- RUNNER:gaussian-kl -->

<!-- RUNNER:attention -->

## 4. 扩散、ODE与SDE绘图

图形直接显示在输出下方。可修改噪声日程、步长和轨迹数量，再解释数值结果与理论是否一致。浏览器端这些例题使用NumPy；模型训练请进入后面的PyTorch Notebook。

<!-- RUNNER:diffusion -->

<!-- RUNNER:sde -->

## 5. SciPy与SymPy

指派例题的最优总代价应为5。符号求导用于核对解析结果，不能代替理解链式法则。

<!-- RUNNER:assignment -->

<!-- RUNNER:symbolic -->

## 6. PyTorch、TensorFlow与完整实验

每份Notebook包含中文步骤、环境检查、代码、结果解释和练习。点击后在Colab按从上到下的顺序运行；需要保存修改时先“在云端硬盘中保存副本”。如需GPU，在“运行时→更改运行时类型”选择可用GPU。首次可能需要下载数据或权重。

<!-- NOTEBOOKS -->

CPU也可完成基础Notebook；DDPM与DINO建议有可用GPU再做完整实验。模型Notebook会下载固定版本课程代码，并校验SHA256；它们没有启动隐藏服务器，也不把Colab当作本站后台。免费GPU、会话时长和网络可达性不保证；不方便使用Colab时可下载Notebook在本地Jupyter运行。[Colab官方说明](https://research.google.com/colaboratory/faq.html)

运行结果保存在临时环境中，断线或重置后可能丢失；可使用Notebook最后的打包单元保存训练输出，然后从Colab文件面板下载ZIP。不要把教程实测数字当作你自己的实验结果。

## 7. 自己Fork和修改

数学示例位于 `data/runner_examples.json`，正文用 `<!-- RUNNER:示例ID -->` 插入；修改后运行 `python scripts/build.py`。Notebook位于 `notebooks/`，可以直接编辑。`scripts/make_notebooks.py` 用于重新生成初始教学版本，会覆盖同名Notebook；通常修改Notebook后只构建网页即可。

Fork后若想让Colab打开自己的版本，将 `data/site.json` 的repository改为自己的“用户名/仓库名”，branch改为实际分支，再重新构建并推送。模型Notebook内的源码地址独立固定在已发布提交；如果也修改训练实现，应在自己的仓库提交后重新生成Notebook并核对源码哈希，再提交生成文件。

实现参考：[Pyodide固定版本的Worker教学](https://pyodide.org/en/0.27.7/usage/webworker.html)、[标准输出接口](https://pyodide.org/en/0.27.7/usage/streams.html)、[TensorFlow官方基础教程](https://www.tensorflow.org/tutorials/quickstart/beginner)。Pyodide运行时与包从公开CDN加载，项目未捆绑完整浏览器Python，因此不承诺首次离线运行。

## 8. 已完成的运行验证

2026-10-28：8个数学示例均在真实浏览器运行过，验证文本/PNG输出、异常提示、torch引导、停止无限循环与再次运行。7份Notebook共39个代码单元按默认配置在本地Jupyter逐格执行通过，包括真实FashionMNIST训练和官方DINOv2预训练/随机骨干对照。Notebook验证环境是本地Python 3.9，未把本地执行记录表述为Colab执行；不同Colab运行时的版本、速度和GPU资源可能不同。[查看验证摘要](evidence/notebook-validation.json)
