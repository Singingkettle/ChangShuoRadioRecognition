# PET-CGDNN — An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation

[English](README.md) | 简体中文

> F. Zhang et al., "An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation", *IEEE Commun. Lett. (2021)*.
> [https://ieeexplore.ieee.org/abstract/document/9507514](https://ieeexplore.ieee.org/abstract/document/9507514)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`petcgdnn`**
（即 `configs/petcgdnn/`）。

## 方法简述

参数估计变换（PET）先旋转 I/Q，再送入紧凑 CGDNN 分类器。Q 旋转符号与 TF 一致。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/petcgdnn.py::PETCGDNN` |
| Train / test configs | `configs/petcgdnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q + PET rotation |

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
python tools/train.py configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 60.26 / 60.00 | 90.43 / 89.00 | `pass` |
| RML2016.10B | 63.80 / 63.00 | 93.52 / 92.00 | `pass` |
| RML2018.01A | 61.24 / 60.00 | 95.69 / 95.00 | `pass` |
| HisarMod | 67.35 / 75.00 | 90.68 / 99.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

RML 通过。Hisar 失败；划分已是官方协议。

