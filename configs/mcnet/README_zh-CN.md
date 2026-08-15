# MCNET — MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification

[English](README.md) | 简体中文

> T. Huynh-The et al., "MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification", *IEEE Commun. Lett. (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/8963964](https://ieeexplore.ieee.org/abstract/document/8963964)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`mcnet`**
（即 `configs/mcnet/`）。

## 方法简述

带 M-block 的高效 CNN（MCNet），用于信道损伤下的稳健 AMC。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/mcnet.py::MCNet` |
| Train / test configs | `configs/mcnet/` |
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
python tools/train.py configs/mcnet/mcnet_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/mcnet/mcnet_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.04 / 58.00 | 84.59 / 82.00 | `pass` |
| RML2016.10B | 62.41 / 62.00 | 91.41 / 87.00 | `pass` |
| RML2018.01A | 58.43 / 55.00 | 92.78 / 92.00 | `pass` |
| HisarMod | 56.59 / 70.00 | 79.59 / 97.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

RML 通过。Hisar 是已知难题（DSP 综述 Table 4 强调收敛差）；不要继续围攻 L2+top1 的 wave17 克隆。

