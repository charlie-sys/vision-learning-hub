# -*- coding: utf-8 -*-
"""Author portable, clean notebooks. No network or training during generation."""
from pathlib import Path
import hashlib
import json
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"
DEST.mkdir(exist_ok=True)
SITE = json.loads((ROOT / "data/site.json").read_text(encoding="utf-8"))
# Pin the source that is currently committed; source bytes are also verified.
REF = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
SOURCE_HASH = hashlib.sha256(subprocess.check_output(["git", "show", f"{REF}:examples/course_labs.py"], cwd=ROOT)).hexdigest()


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    item = {"cell_type": kind, "id": hashlib.sha256((kind+source).encode()).hexdigest()[:12],
            "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def md(text): return cell("markdown", text)
def code(text): return cell("code", text)


def write(name, cells, gpu=False):
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "colab": {"name": name + ".ipynb", "provenance": []}}
    if gpu: metadata["accelerator"] = "GPU"
    result = {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": cells}
    (DEST / (name+".ipynb")).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


intro = md("""
    由 [Vision Learning Hub](https://charlie-sys.github.io/vision-learning-hub/) 提供。
    在 Colab 登录自己的Google账号，按从上到下的顺序运行；修改前可“在云端硬盘中保存副本”。
    若需要GPU，在“运行时 → 更改运行时类型”中选择可用GPU。免费GPU不保证随时可用，断线后临时文件可能丢失，请下载结果。
    本Notebook的结果属于你自己的运行，不应把教程实测数值直接当作本次成绩。
""")
write("00_numpy_math", [md("# NumPy / SciPy / SymPy 数学实验"), intro,
    md("## 1. 环境检查\n依赖由当前Python环境提供；Colab通常已包含以下数学库。"),
    code("import sys\nimport numpy as np\nimport scipy\nimport sympy as sp\nimport matplotlib.pyplot as plt\nprint(sys.version)\nprint('numpy:', np.__version__, 'scipy:', scipy.__version__, 'sympy:', sp.__version__)"),
    md("## 2. 手算与数值检查\n先预测输出，再运行；改变参数比较解析梯度与有限差分。"),
    code(json.loads((ROOT/"data/runner_examples.json").read_text(encoding="utf-8"))[1]["code"]),
    md("## 3. 高斯分布与KL\n样本均值/方差只近似理论值，增加样本量观察误差。"),
    code(json.loads((ROOT/"data/runner_examples.json").read_text(encoding="utf-8"))[2]["code"]),
    md("## 4. SciPy全局指派"),
    code(json.loads((ROOT/"data/runner_examples.json").read_text(encoding="utf-8"))[6]["code"]),
    md("## 5. 符号求导"),
    code(json.loads((ROOT/"data/runner_examples.json").read_text(encoding="utf-8"))[7]["code"]),
    md("## 6. Brownian方差与Euler误差\n图像应接近理论线，但不会完全重合。"),
    code(json.loads((ROOT/"data/runner_examples.json").read_text(encoding="utf-8"))[5]["code"]),
    md("## 练习\n将时间步缩小一半，比较Euler误差；解释SDE噪声为何乘步长平方根。")])

write("01_pytorch_basics", [md("# PyTorch：张量、自动微分与训练"), intro,
    md("## 1. 库与设备\nColab预装框架；若在本地运行，先按PyTorch官方安装向导安装与硬件匹配的版本。"),
    code("import torch\nimport matplotlib.pyplot as plt\nprint('torch:', torch.__version__)\ndevice = 'cuda' if torch.cuda.is_available() else 'cpu'\nprint('device:', device)\ntorch.manual_seed(7)"),
    md("## 2. 自动微分\n例题预期loss=2，w的梯度=4。注意每次backward会累加梯度。"),
    code("w = torch.tensor(3., requires_grad=True, device=device)\nx, b, target = 2., 1., 5.\nloss = (w*x+b-target)**2/2\nloss.backward()\nprint('loss:', loss.item(), 'gradient:', w.grad.item())\nassert torch.isclose(w.grad, torch.tensor(4., device=device))"),
    md("## 3. 完整线性回归\n目标为y=3x+1，添加少量噪声。训练和验证数据独立生成。"),
    code("""
        x_train = torch.randn(256, 1, device=device)
        y_train = 3*x_train + 1 + .1*torch.randn_like(x_train)
        x_val = torch.randn(128, 1, device=device)
        y_val = 3*x_val + 1 + .1*torch.randn_like(x_val)
        model = torch.nn.Linear(1, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        history = []
        for step in range(100):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x_train), y_train)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        model.eval()
        with torch.inference_mode():
            val_loss = torch.nn.functional.mse_loss(model(x_val), y_val).item()
        print('learned weight/bias:', model.weight.item(), model.bias.item())
        print('validation MSE:', val_loss)
        assert val_loss < .1
        plt.plot(history)
        plt.xlabel('optimization step')
        plt.ylabel('training MSE')
        plt.show()
    """),
    md("## 4. 保存与恢复\n训练输出应及时下载，运行时重置会丢失临时文件。"),
    code("torch.save(model.state_dict(), 'linear_torch.pt')\nrestored = torch.nn.Linear(1, 1).to(device)\nrestored.load_state_dict(torch.load('linear_torch.pt', map_location=device, weights_only=True))\nwith torch.inference_mode():\n    torch.testing.assert_close(model(x_val), restored(x_val))\nprint('checkpoint reload passed')"),
    md("## 练习\n将学习率改为1，观察是否发散；与0.01比较收敛速度。不能用验证数据参与更新。")])

write("02_tensorflow_basics", [md("# TensorFlow：GradientTape 与 Keras训练"), intro,
    md("## 1. 环境\nTensorFlow是本Notebook的Python框架，不是浏览器TensorFlow.js。GPU不可用时CPU也可完成。"),
    code("import numpy as np\nimport tensorflow as tf\nimport matplotlib.pyplot as plt\nprint('tensorflow:', tf.__version__)\nprint('GPUs:', tf.config.list_physical_devices('GPU'))\ntf.keras.utils.set_random_seed(7)"),
    md("## 2. 同一个梯度例题\n应与PyTorch和手算一致：loss=2，dw=4。"),
    code("w = tf.Variable(3.)\nwith tf.GradientTape() as tape:\n    loss = (w*2+1-5)**2/2\ngrad = tape.gradient(loss, w)\nprint('loss:', float(loss.numpy()), 'gradient:', float(grad.numpy()))\nnp.testing.assert_allclose(grad.numpy(), 4.)"),
    md("## 3. 合成二分类数据\n两团二维高斯样本，标签不需要下载。训练与验证独立生成；这不是视觉性能基准。"),
    code("""
        rng = np.random.default_rng(7)
        def data(n):
            y = rng.integers(0, 2, n, dtype=np.int32)
            x = rng.normal(0, .5, (n, 2)).astype('float32') + (2*y[:, None]-1).astype('float32')
            return x, y
        x_train, y_train = data(256)
        x_val, y_val = data(128)
        print(x_train.shape, y_train.shape)
    """),
    md("## 4. Keras模型与显式训练循环\n输出logits，交叉熵设置from_logits=True；只更新模型的trainable_variables。"),
    code("""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2),
        ])
        optimizer = tf.keras.optimizers.Adam(.03)
        objective = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        history = []
        for step in range(60):
            with tf.GradientTape() as tape:
                logits = model(x_train, training=True)
                loss = objective(y_train, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            history.append(float(loss.numpy()))
        logits = model(x_val, training=False)
        predictions = tf.argmax(logits, axis=-1).numpy()
        accuracy = np.mean(predictions == y_val)
        print('validation accuracy:', accuracy)
        assert accuracy > .9
        plt.plot(history)
        plt.xlabel('optimization step')
        plt.ylabel('cross entropy')
        plt.show()
    """),
    md("## 5. 保存与恢复"),
    code("model.save('tensorflow_classifier.keras')\nrestored = tf.keras.models.load_model('tensorflow_classifier.keras')\nnp.testing.assert_allclose(model(x_val, training=False).numpy(), restored(x_val, training=False).numpy(), rtol=1e-5, atol=1e-6)\nprint('Keras model reload passed')"),
    md("## 练习\n改变数据噪声0.5→1.5，观察错误样本；对照PyTorch里梯度清零与GradientTape的使用差异。\n参考：[TensorFlow官方快速入门](https://www.tensorflow.org/tutorials/quickstart/beginner)。")])

bootstrap = code(f"""
    from pathlib import Path
    import hashlib, urllib.request, importlib.util, argparse, json
    source_url = 'https://raw.githubusercontent.com/{SITE['repository']}/{REF}/examples/course_labs.py'
    folder = Path('vision_learning_runtime')
    folder.mkdir(exist_ok=True)
    source = folder / 'course_labs.py'
    urllib.request.urlretrieve(source_url, source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '{SOURCE_HASH}', 'Source checksum differs; stop and verify the source.'
    spec = importlib.util.spec_from_file_location('course_labs', source)
    labs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labs)
    print('Verified course source: {REF}')
""")

for number, mode, epochs in [(3, 'cnn', 3), (4, 'vae', 6), (5, 'ddpm', 10), (6, 'probe', 20)]:
    title = {'cnn':'CNN图像分类', 'vae':'VAE编码、重建与先验采样', 'ddpm':'DDPM训练与完整反向采样', 'probe':'DINOv2冻结特征与随机骨干对照'}[mode]
    cells = [md(f"# {title}"), intro,
        md("## 1. 加载经过版本和SHA256校验的课程代码\n首次需要网络。下载源为本课程公开GitHub仓库；可在运行前查看链接内容。若依赖缺失，请按实验网页安装匹配的torch/torchvision。"), bootstrap,
        md("## 2. 配置\n先保留默认设置跑通。CPU较慢时减少样本和轮数；synthetic只用于流程检查，不能替代FashionMNIST或真实DINO结果。"),
        code(f"""
            import torch
            from IPython.display import display, Image, SVG
            args = argparse.Namespace(
                mode='{mode}', dataset='fashion', seed=7,
                data_dir=str(folder / 'datasets'), output=str(folder / 'runs-{mode}'),
                train_size={1024 if mode == 'probe' else 2048}, val_size={256 if mode == 'probe' else 512},
                epochs={epochs}, batch_size={16 if mode == 'probe' else 64}, lr={.01 if mode == 'probe' else .001},
                device='cuda' if torch.cuda.is_available() else 'cpu', resume=None,
                latent_dim=16, beta=1., steps=200, encoder='dinov2',
                hub_ref=labs.DINO_REF, hub_dir=None, random_baseline=True, evaluate_test=False,
            )
            Path(args.output).mkdir(parents=True, exist_ok=True)
            labs.seed_all(args.seed)
            print('torch:', torch.__version__, 'device:', args.device)
            print(vars(args))
        """),
        md("## 3. 数据划分\n仅使用官方训练数据中的不重叠训练/验证子集；不读取官方测试集。DINO的特征标准化仅用训练特征统计。"),
        code("train, val, test, metadata = labs.data_splits(args)\nx, y = next(iter(labs.loader(train, args)))\nprint('batch:', tuple(x.shape), tuple(y.shape))\nprint('split metadata:', metadata)"),
        md("## 4. 训练与验证\n阅读网页中对应损失与流程说明，再运行。最佳模型根据验证集选择，不能把验证结果当作测试成绩。DINO只训练线性头，不训练预训练骨干。")]
    trainer = "labs.train_cnn(args, train, val, test)" if mode == 'cnn' else "labs.linear_probe(args, train, val, test)" if mode == 'probe' else "labs.train_generative_or_cnn(args, train, val, test, metadata)"
    cells.append(code(f"history, result = {trainer}\nprint(json.dumps(result, indent=2))\nmetadata.update(config=vars(args), results=result, history=history, torch=str(torch.__version__))\n(Path(args.output) / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')\nlabs.write_curve(history, Path(args.output) / 'curves.svg')"))
    cells += [md("## 5. 查看曲线、图像和检查点\n图像显示当前运行结果。小模型可能产生模糊或破碎样本，请记录失败情况，不仅展示最好一张。"),
        code("display(SVG(filename=str(Path(args.output) / 'curves.svg')))\nfor image in sorted(Path(args.output).glob('*.png')):\n    print(image.name)\n    display(Image(filename=str(image)))\nprint('Artifacts:', [p.name for p in Path(args.output).iterdir()])"),
        md("## 6. 打包结果\n下载ZIP前确认内容；文件含训练权重和配置，不包含原始数据集。Colab左侧文件面板可以下载下方生成的ZIP。"),
        code(f"import shutil\narchive = shutil.make_archive('results_{mode}', 'zip', args.output)\nprint('Download this file using the notebook file panel:', archive)"),
        md("## 练习与验收\n" + {'cnn':'比较学习率0.001与0.01；解释混淆矩阵中最常见的错误。', 'vae':'比较beta为0.1/1/4，分别记录重建项与KL，不直接比较不同beta的总损失。', 'ddpm':'检查噪声预测误差是否低于零预测器，查看完整反向采样；8步冒烟不能验证生成质量。', 'probe':'比较预训练与随机骨干；确认两者同划分、同预算、骨干冻结。tiny只能做冒烟测试。'}[mode] + "\n\n修改seed为17、27并更换输出目录，报告均值和样本标准差。[完整教学与排错](https://charlie-sys.github.io/vision-learning-hub/practice.html)。")]
    write(f"{number:02d}_{mode}", cells, gpu=True)

print(f"Generated {len(list(DEST.glob('*.ipynb')))} clean notebooks; source ref {REF}")
