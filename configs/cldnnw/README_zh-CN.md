# CLDNNW — Deep architectures for modulation recognition

[English](README.md) | 简体中文

> N. E. West and T. O’Shea, "Deep architectures for modulation recognition", *IEEE DySPAN (2017)*.
> [https://ieeexplore.ieee.org/abstract/document/7920754](https://ieeexplore.ieee.org/abstract/document/7920754)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`cldnnw`**
（即 `configs/cldnnw/`）。

## 方法简述

West/O’Shea 的 CLDNN：三个 (1×8) 卷积加 dropout，特征拼接后送入 LSTM，再接全连接分类器。CSRR 通过 `use_zero_pad=True` 恢复 TF 的 `ZeroPadding2D((0,2))`（旧的无 padding checkpoint 设为 False）。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/cldnn.py::CLDNNW` |
| Train / test configs | `configs/cldnnw/` |
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
python tools/train.py configs/cldnnw/cldnnw_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/cldnnw/cldnnw_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 56.54 / 57.00 | 84.05 / 85.00 | `pass` |
| RML2016.10B | 60.35 / 62.00 | 88.05 / 89.00 | `pass` |
| RML2018.01A | 37.19 / 55.00 | 53.33 / 88.00 | `fail` |
| HisarMod | 66.54 / 75.00 | 96.17 / 98.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

对齐 ZeroPad 后，10A/10B 在近似容差下通过。RML2018.01A 与 Hisar 仍大幅失败（长序列 / 平台）。不要再围攻相同的 wave17 Hisar 循环。

