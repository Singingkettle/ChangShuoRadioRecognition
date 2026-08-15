# CGDNet — CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition

[English](README.md) | 简体中文

> Y. Wang et al., "CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition", *IEEE Commun. Lett.* (2021).
> [https://ieeexplore.ieee.org/abstract/document/9349627](https://ieeexplore.ieee.org/abstract/document/9349627)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`cgdnet`**
（即 `configs/cgdnet/`）。

## 方法简述

紧凑的 CNN–GRU 混合网络：卷积前端提取局部 I/Q 特征，门控循环单元再聚合时序上下文以预测调制类别。CSRR 移植了 AMR-Benchmark 的 CGDNet 拓扑，并对长序列（2018 / Hisar）固定 `frame_length`。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/cgdnet.py::CGDNet` |
| Train / test configs | `configs/cgdnet/` |
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
python tools/train.py configs/cgdnet/cgdnet_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/cgdnet/cgdnet_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 56.96 / 58.00 | 83.66 / 83.00 | `pass` |
| RML2016.10B | 61.15 / 62.00 | 89.49 / 88.00 | `pass` |
| RML2018.01A | 35.87 / 57.00 | 51.67 / 92.00 | `fail` |
| HisarMod | 71.25 / (CSRR-only) | 95.69 / (CSRR-only) | `measured` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

RML2018.01A 仍是大幅失败（长序列塌缩）。Hisar 仅为 CSRR 实测。默认 RML 划分是 CSRR 的 50/10/40，部分公开移植用 6:2:2。

