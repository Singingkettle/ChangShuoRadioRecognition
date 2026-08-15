# CNN4 — Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels

[English](README.md) | 简体中文

> K. Youssef et al. / AMR-Benchmark CNN2 multipath, "Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels", *IEEE VTC (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/9128408](https://ieeexplore.ieee.org/abstract/document/9128408)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`cnn4`**
（即 `configs/cnn4/`）。

## 方法简述

面向多径的 CNN（CSRR `CNN4`），卷积核固定为 (2,8) 以对齐 AMR-Benchmark 的多径 CNN2 移植。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/cnn4.py::CNN4` |
| Train / test configs | `configs/cnn4/` |
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
python tools/train.py configs/cnn4/cnn4_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/cnn4/cnn4_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.57 / 58.00 | 83.89 / 80.00 | `pass` |
| RML2016.10B | 61.83 / 63.00 | 89.61 / 84.00 | `pass` |
| RML2018.01A | 54.55 / 55.00 | 84.57 / 91.00 | `fail` |
| HisarMod | 75.08 / 70.00 | 99.81 / 98.00 | `pass` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

2018 总体接近；峰值仍不足。从官方冠军做 SelfNormalize 微调会把验证集打崩，已放弃。

