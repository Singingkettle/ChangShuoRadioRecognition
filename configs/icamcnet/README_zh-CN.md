# IC-AMCNet — CNN-Based Automatic Modulation Classification for Beyond 5G Communications

[English](README.md) | 简体中文

> A. P. Hermawan et al., "CNN-Based Automatic Modulation Classification for Beyond 5G Communications", *IEEE Commun. Lett. (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/8977561](https://ieeexplore.ieee.org/abstract/document/8977561)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`icamcnet`**
（即 `configs/icamcnet/`）。

## 方法简述

带高斯噪声正则的深度 CNN（IC-AMCNet）。在长帧（2018/Hisar）上参数量很大。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/icamcnet.py::ICAMCNet` |
| Train / test configs | `configs/icamcnet/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q |

## 数据

DeepSig RML 的 JSON 位于 `data/ModulationClassification/DeepSig/`，本仓库采用
**50/10/40** 划分（`train.json` / `validation.json` / `test.json`）。部分公开的
Keras 移植按每个（调制，信噪比）做 **6:2:2**；个别数据集上的小幅总体差距可能
来自这一划分差异。

HisarMod 的 JSON 位于 `data/ModulationClassification/Hisar/HisarMod2019.1/`，
已经遵循**官方 Test + Train 80/20** 协议（约 416k / 104k / 260k）。不要把
Hisar 上的残差归因于 50/10/40 划分。

## 训练 / 评测

```bash
# 训练（默认 work_dir 在 work_dirs/ 下）
python tools/train.py configs/icamcnet/icamcnet_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/icamcnet/icamcnet_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 56.79 / 57.00 | 85.07 / 83.00 | `pass` |
| RML2016.10B | 61.66 / 62.00 | 91.67 / 87.00 | `pass` |
| RML2018.01A | 59.49 / 58.00 | 95.13 / 92.00 | `pass` |
| HisarMod | 83.41 / 80.00 | 98.58 / 100.00 | `pass` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

四个跟踪数据集全部通过（Hisar 峰值按接近匹配 98.58 ≥ 98.5）。冲峰值 100 的 ES 循环已关闭。

