## 项目简介

[English](README.md) | 简体中文

CSRR（ChangShuoRadioRecognition）是一个基于 PyTorch 和 [MMEngine](https://github.com/open-mmlab/mmengine) 的开源自动调制识别（AMC）工具箱。

### 主要特性

- **配置驱动**：所有实验通过 Python 配置文件管理
- **多种 AMC 方法**：支持 20+ 种最先进的 AMC 算法
- **性能分析**：内置工具可生成论文级别的图表
- **最小依赖**：仅依赖 MMEngine，无需其他 MM 系列框架

## 支持的方法

| 方法 | 论文 |
|------|------|
| [CGDNet](configs/cgdnet) | [CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/9349627) |
| [CLDNNW](configs/cldnnw) | [Deep architectures for modulation recognition](https://ieeexplore.ieee.org/abstract/document/7920754) |
| [CLDNNL](configs/cldnnl) | [Deep Neural Network Architectures for Modulation Classification](https://ieeexplore.ieee.org/document/8335483) |
| [CNN1DPF](configs/cnn1dpf) | [Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks](https://lirias.kuleuven.be/retrieve/546033) |
| [CNN2](configs/cnn2) | [Convolutional Radio Modulation Recognition Networks](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16) |
| [CNN4](configs/cnn4) | [Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels](https://ieeexplore.ieee.org/abstract/document/9128408) |
| [DAE](configs/dae) | [Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder](https://ieeexplore.ieee.org/abstract/document/9487492) |
| [DensCNN](configs/denscnn) | [Deep Neural Network Architectures for Modulation Classification](https://ieeexplore.ieee.org/document/8335483) |
| [DSCLDNN](configs/dscldnn) | [Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure](https://ieeexplore.ieee.org/document/9220797) |
| [FastMLDNN](configs/fastmldnn) | [A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/10239249) |
| [GRU2](configs/gru2) | [Automatic modulation classification using recurrent neural networks](https://ieeexplore.ieee.org/abstract/document/8322633) |
| [HCGDNN](configs/hcgdnn) | [A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/document/9764618) |
| [LSTM2](configs/lstm2) | [Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors](https://ieeexplore.ieee.org/abstract/document/8357902) |
| [IC-AMCNet](configs/icamcnet) | [CNN-Based Automatic Modulation Classification for Beyond 5G Communications](https://ieeexplore.ieee.org/abstract/document/8977561) |
| [MCformer](configs/mcformer) | [MCformer: A Transformer Based Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/9685815) |
| [MCLDNN](configs/mcldnn) | [A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/9106397) |
| [MCNET](configs/mcnet) | [MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/8963964) |
| [MLDNN](configs/mldnn) | [Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification](https://ieeexplore.ieee.org/document/9462447) |
| [PET-CGDNN](configs/petcgdnn) | [An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation](https://ieeexplore.ieee.org/abstract/document/9507514) |
| [TRN](configs/trn) | [Signal Modulation Classification Based on the Transformer Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340) |

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.8
- MMEngine >= 0.7.0

### 安装步骤

```bash
# 创建 conda 环境
conda create -n csrr python=3.10 -y
conda activate csrr

# 安装 PyTorch（根据你的 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装 MMEngine
pip install mmengine

# 安装 CSRR
git clone https://github.com/Singingkettle/ChangShuoRadioRecognition.git
cd ChangShuoRadioRecognition
pip install -e .
```

## 快速开始

### 1. 模型训练

```bash
# 在 DeepSig RadioML.2016.10A 数据集上训练 CNN2
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py

# 指定工作目录
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py --work-dir work_dirs/my_exp

# 覆盖配置选项
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py \
    --cfg-options train_cfg.max_epochs=100
```

### 2. 模型测试

训练完成后，生成用于性能分析的预测结果：

```bash
# 测试并保存预测结果
python tools/test.py configs/cnn2/cnn2_iq-deepsig-201610A.py \
    work_dirs/cnn2_iq-deepsig-201610A/best_accuracy_top1_epoch_*.pth \
    --work-dir work_dirs/cnn2_iq-deepsig-201610A
```

这会保存 `paper.pkl` 文件，包含预测结果、真实标签和信噪比值。

### 3. 性能分析（绘图）

通过比较多个已训练模型来生成论文级别的图表：

**步骤 1**：在 `configs/_base_/performance_info/amc.py` 中配置方法注册表：

```python
info = dict(
    work_dir='work_dirs',
    methods={
        'CNN2': 0,    # 索引决定颜色/标记
        'CNN4': 1,
        'DensCNN': 2,
    },
    publish=dict(
        deepsig201610A=dict(
            CNN2='cnn2_iq-deepsig-201610A',      # 方法名 -> 工作目录名
            CNN4='cnn4_iq-deepsig-201610A',
            DensCNN='denscnn_iq-deepsig-201610A',
        ),
    ),
)
```

**步骤 2**：创建绘图配置文件（如 `configs/cnn2/cnn2_plot.py`）：

```python
_base_ = ['../_base_/performance_info/amc.py']

performance = dict(
    type='Classification',
    Figures=[
        dict(
            type='SNRVsAccuracy',
            dataset=dict(
                deepsig201610A=dict(
                    comparison=['CNN2', 'CNN4', 'DensCNN'],
                ),
            ),
        ),
        dict(
            type='ConfusionMap',
            dataset=dict(deepsig201610A=['CNN2']),
        ),
    ],
)
```

**步骤 3**：生成图表：

```bash
python tools/analyze.py configs/cnn2/cnn2_plot.py
```

输出的 PDF 文件保存在 `work_dirs/performance/`。

#### 支持的图表类型

| 类型 | 描述 |
|------|------|
| `SNRVsAccuracy` | 准确率 vs 信噪比曲线（折线图 + 雷达图） |
| `ClassVsF1ScoreWithSNR` | 每个信噪比下各类别的 F1 分数 |
| `ConfusionMap` | 每个信噪比下的混淆矩阵 |

## 项目结构

```
ChangShuoRadioRecognition/
├── configs/                 # 模型和实验配置
│   ├── _base_/             # 基础配置（数据集、调度等）
│   ├── cnn2/               # CNN2 模型配置
│   └── ...
├── csrr/                   # 核心库
│   ├── datasets/           # 数据集类
│   ├── models/             # 模型实现
│   ├── engine/             # 训练钩子
│   ├── evaluation/         # 评估指标
│   ├── performance/        # 绘图和分析工具
│   └── ...
├── tools/                  # 命令行工具
│   ├── train.py           # 训练脚本
│   ├── test.py            # 测试/推理脚本
│   └── analyze.py         # 性能分析脚本
└── work_dirs/             # 输出目录（训练时创建）
```

## 引用

如果您在研究中使用了本工具箱，请引用：

```bibtex
@article{chang2021multi,
  title={Multi-task learning based deep neural network for automatic modulation classification},
  author={Chang, Shuo and Huang, Sai and Zhang, Ruiyun and Feng, Zhiyong and Liu, Liang},
  journal={IEEE Internet of Things Journal},
  year={2021},
  publisher={IEEE}
}

@article{chang2022hcgdnn,
  author={Chang, Shuo and Zhang, Ruiyun and Ji, Kejia and Huang, Sai and Feng, Zhiyong},
  journal={IEEE Transactions on Wireless Communications}, 
  title={A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification}, 
  year={2022},
  doi={10.1109/TWC.2022.3168884}
}

@article{chang2023fastmldnn,
  title={A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification},
  author={Chang, Shuo and others},
  journal={IEEE Transactions},
  year={2023}
}
```

## 许可证

本项目采用 [Apache 2.0 许可证](LICENSE)。

## 联系方式

本项目由常硕（[@Singingkettle](https://github.com/Singingkettle)）维护。
