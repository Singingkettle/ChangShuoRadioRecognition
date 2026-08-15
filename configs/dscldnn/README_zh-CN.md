# DSCLDNN — Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure

[English](README.md) | 简体中文

> Z. Zhang et al., "Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure", *IEEE Access (2020)*.
> [https://ieeexplore.ieee.org/document/9220797](https://ieeexplore.ieee.org/document/9220797)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`dscldnn`**
（即 `configs/dscldnn/`）。

## 方法简述

双流 CNN–LSTM：一路处理 I/Q，一路处理幅度/相位，融合后再分类。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/dscldnn.py::DSCLDNN` |
| Train / test configs | `configs/dscldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P + I/Q dual stream |

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
python tools/train.py configs/dscldnn/dscldnn_ap-iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/dscldnn/dscldnn_ap-iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

_本方法没有 AMR-Benchmark 跟踪行。请用下面的训练命令本地记录指标。_

## 已记录的偏差 / 说明

不在已关闭的 AMR-Benchmark 跟踪矩阵中；配置仅为完整性提供。请使用 `configs/dscldnn/` 下的根配置。

