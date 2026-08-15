# LSTM2 — Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors

[English](README.md) | 简体中文

> S. Rajendran et al., "Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors", *IEEE Trans. Cogn. Commun. Netw. (2018)*.
> [https://ieeexplore.ieee.org/abstract/document/8357902](https://ieeexplore.ieee.org/abstract/document/8357902)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`lstm2`**
（即 `configs/lstm2/`）。

## 方法简述

在幅度/相位（L×F）上的两层 LSTM。TF 与 CSRR 都用 A/P — 原始 I/Q 会把准确率打崩。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/lstm2.py::LSTM2` |
| Train / test configs | `configs/lstm2/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P |

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
python tools/train.py configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 63.53 / 58.00 | 93.73 / 87.00 | `pass` |
| RML2016.10B | 63.94 / 64.00 | 93.66 / 94.00 | `pass` |
| RML2018.01A | 62.30 / 60.00 | 97.02 / 98.00 | `pass` |
| HisarMod | 69.91 / 73.00 | 97.00 / 98.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

RML 通过。Hisar 总体在打磨后仍偏低；Hisar 划分已是官方协议。

