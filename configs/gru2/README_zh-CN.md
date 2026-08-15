# GRU2 — Automatic modulation classification using recurrent neural networks

[English](README.md) | 简体中文

> D. Hong et al. / AMR-Benchmark GRU, "Automatic modulation classification using recurrent neural networks", *IEEE ICSPCC (2017)*.
> [https://ieeexplore.ieee.org/abstract/document/8322633](https://ieeexplore.ieee.org/abstract/document/8322633)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`gru2`**
（即 `configs/gru2/`）。

## 方法简述

在重排后的 I/Q（`L×F`）上的两层 GRU 分类器。对齐 DSP 综述所用的 AMR-Benchmark GRU Keras 路径。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/gru2.py::GRU2` |
| Train / test configs | `configs/gru2/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q (L×F reshape) |

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
python tools/train.py configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.84 / 58.00 | 85.95 / 85.00 | `pass` |
| RML2016.10B | 64.53 / 63.00 | 93.50 / 91.00 | `pass` |
| RML2018.01A | 61.95 / 59.00 | 96.37 / 95.00 | `pass` |
| HisarMod | 69.34 / 73.00 | 97.02 / 98.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

全部 RML 通过。Hisar 总体在 wave17 平台期后仍低于近似门槛；划分已是官方 Test + Train 80/20。

