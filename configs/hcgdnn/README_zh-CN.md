# HCGDNN — A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification

[English](README.md) | 简体中文

> S. Chang et al., "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification", *IEEE Wireless Commun. Lett. (2022)*.
> [https://ieeexplore.ieee.org/document/9764618](https://ieeexplore.ieee.org/document/9764618)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`hcgdnn`**
（即 `configs/hcgdnn/`）。

## 方法简述

自有方法 A 档：卷积门控网络上的层次分类头。论文原生 50/10/40。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/hcgdnn.py::HCGDNN` |
| Train / test configs | `configs/hcgdnn/` |
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
python tools/train.py configs/hcgdnn/hcgdnn_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/hcgdnn/hcgdnn_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 63.43 / 64.90 | 93.36 / 93.00 | `pass` |
| RML2016.10B | 65.04 / (CSRR-only) | 93.71 / (CSRR-only) | `measured` |
| RML2018.01A | 58.72 / (CSRR-only) | 93.52 / (CSRR-only) | `measured` |
| HisarMod | 57.39 / (CSRR-only) | 70.16 / (CSRR-only) | `measured` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

10A 跟踪通过，63.43/93.36 对 64.9/93。其他数据集仅为实测。论文精确围攻已关闭。

