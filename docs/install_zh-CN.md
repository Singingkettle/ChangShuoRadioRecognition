## 安装

[English](install.md) | 简体中文

### 环境要求

- Ubuntu 22.04 LTS
- Python 3.10
- PyTorch 2.0.1
- CUDA 11.8

### 安装 ChangShuoRadioRecognition

a. 将 g++/gcc 升级到 5 以上，以便编译 C++ 文件。

b. 创建并激活 conda 虚拟环境。

```shell
conda create -n ChangShuoRadioRecognition python=3.10 -y
conda activate ChangShuoRadioRecognition
```

c. 安装 PyTorch 与 torchvision。
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

d. 安装 mmdet，用于无线电检测与识别。

```shell
pip install -U openmim
mim install mmengine
mim install mmdet
```

e. 克隆 ChangShuoRadioRecognition 仓库。

```shell
git clone https://github.com/ChangShuoRadioRecognition.git
cd ChangShuoRadioRecognition
```

f. 安装构建依赖，再安装 ChangShuoRadioRecognition。

```shell
mim install -e .  # or "python setup_backup.py develop"
```

g. 为 matplotlib 安装 “Times New Roman” 字体，步骤见
[这篇说明](https://blog.csdn.net/u014712482/article/details/80568540?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control)。

```shell
sudo apt-get update
sudo apt-get install ttf-mscorefonts-installer
rm -rf ~/.cache/matplotlib
```

h. 按 [该链接](https://github.com/CannyLab/tsne-cuda) 安装 tsnecuda。
