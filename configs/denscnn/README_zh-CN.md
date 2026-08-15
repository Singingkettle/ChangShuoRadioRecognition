# DensCNN — Deep Neural Network Architectures for Modulation Classification

[English](README.md) | 简体中文

> X. Liu, D. Yang, and A. El Gamal, "Deep Neural Network Architectures for Modulation Classification", *IEEE Asilomar (2017)*.
> [https://ieeexplore.ieee.org/document/8335483](https://ieeexplore.ieee.org/document/8335483)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`denscnn`**
（即 `configs/denscnn/`）。

## 方法简述

来自 Asilomar 架构论文 / AMR-Benchmark DenseNet 条目的 DenseNet 风格卷积分类器（`DensCNN`）。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/denscnn.py::DensCNN` |
| Train / test configs | `configs/denscnn/` |
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
python tools/train.py configs/denscnn/denscnn_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/denscnn/denscnn_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 58.92 / 57.00 | 90.30 / 83.00 | `pass` |
| RML2016.10B | 60.72 / 62.00 | 91.87 / 87.00 | `pass` |
| RML2018.01A | 53.99 / 58.00 | 89.45 / 92.00 | `fail` |
| HisarMod | 83.85 / 80.00 | 100.00 / 100.00 | `pass` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

10A/10B/Hisar 通过。2018 总体/峰值略低于近似门槛；继续 SelfNormalize 微调会回退。

