# MCLDNN — A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition

[English](README.md) | 简体中文

> J. Xu, C. Yang, et al., "A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition", *IEEE Wireless Commun. Lett. (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/9106397](https://ieeexplore.ieee.org/abstract/document/9106397)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`mcldnn`**
（即 `configs/mcldnn/`）。

## 方法简述

多通道 CNN + LSTM（MCLDNN）。CSRR 的 reshape 对齐 Keras `(L-4, 100)`。作为对照模型，在 50/10/40 下通过全部 RML 集。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/mcldnn.py::MCLDNN` |
| Train / test configs | `configs/mcldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q multi-branch |

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
python tools/train.py configs/mcldnn/mcldnn_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/mcldnn/mcldnn_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 61.75 / 62.00 | 92.45 / 92.05 | `pass` |
| RML2016.10B | 64.65 / 65.00 | 93.87 / 93.00 | `pass` |
| RML2018.01A | 61.56 / 60.00 | 96.83 / 95.00 | `pass` |
| HisarMod | 71.20 / 75.00 | 98.94 / 99.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

Hisar 总体仍偏低；划分已是官方协议。RML 通过是这次移植的对齐对照。

