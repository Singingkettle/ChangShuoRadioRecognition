# MLDNN — Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification

[English](README.md) | 简体中文

> S. Chang et al., "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification", *IEEE Trans. Veh. Technol. (2021)*.
> [https://ieeexplore.ieee.org/document/9462447](https://ieeexplore.ieee.org/document/9462447)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`mldnn`**
（即 `configs/mldnn/`）。

## 方法简述

自有方法 A 档：共享主干、调制（及可选 SNR）多头的多任务 MLDNN。论文原生 50/10/40。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/mldnn.py::MLDNN` |
| Train / test configs | `configs/mldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q + A/P |

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
python tools/train.py configs/mldnn/mldnn_iq-ap-deepsig201610A.py

# 测试一个 checkpoint
python tools/test.py configs/mldnn/mldnn_iq-ap-deepsig201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 62.31 / 62.00 | 92.73 / 92.00 | `pass` |
| RML2016.10B | 65.06 / (CSRR-only) | 93.62 / (CSRR-only) | `measured` |
| RML2018.01A | 57.94 / (CSRR-only) | 90.77 / (CSRR-only) | `measured` |
| HisarMod | 60.06 / (CSRR-only) | 73.63 / (CSRR-only) | `measured` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

10A 是论文精确通过。其他数据集仅为 CSRR 实测。

