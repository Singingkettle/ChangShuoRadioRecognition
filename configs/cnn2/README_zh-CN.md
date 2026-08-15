# CNN2 — Convolutional Radio Modulation Recognition Networks

[English](README.md) | 简体中文

> T. J. O’Shea, J. Corgan, and T. C. Clancy, "Convolutional Radio Modulation Recognition Networks", *EAI IntelliSys / Springer (2016)*.
> [https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`cnn2`**
（即 `configs/cnn2/`）。

## 方法简述

经典的 O’Shea CNN1（CSRR 中为 `CNN2`）：两层卷积（50×1×8）、dropout 与全连接，用于 11 类 RML2016.10A（其他数据集有对应配置）。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/cnn2.py::CNN2` |
| Train / test configs | `configs/cnn2/` |
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
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/cnn2/cnn2_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 63.18 / 59.00 | 81.43 / 79.00 | `pass` |
| RML2016.10B | 56.25 / 64.00 | 81.58 / 85.00 | `fail` |
| RML2018.01A | 42.35 / 58.00 | 65.23 / 92.00 | `fail` |
| HisarMod | 79.74 / 75.00 | 100.00 / 100.00 | `pass` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

10A 与 Hisar 通过。10B 尤其是 2018 失败明显 — 2018 长序列的弱势相对 Fig. 5 读数是结构性的，不是缺 padding。

