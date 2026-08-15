# TRN — Signal Modulation Classification Based on the Transformer Network

[English](README.md) | 简体中文

> J. Cai et al., "Signal Modulation Classification Based on the Transformer Network", *IEEE Trans. Cogn. Commun. Netw. (2022)*.
> [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`trn`**
（即 `configs/trn/`）。

## 方法简述

在星座图 / 类图像输入上做调制分类的 Transformer 网络。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `(image-backbone path via configs/trn)` |
| Train / test configs | `configs/trn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | Image / constellation |

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
python tools/train.py configs/trn/trn_img-deepsig201610A.py

# 测试一个 checkpoint
python tools/test.py configs/trn/trn_img-deepsig201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

_本方法没有 AMR-Benchmark 跟踪行。请用下面的训练命令本地记录指标。_

## 已记录的偏差 / 说明

不在已关闭的 AMR I/Q 跟踪矩阵中；图像流水线配置在 `configs/trn/`。

