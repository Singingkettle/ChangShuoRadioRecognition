# FastMLDNN — A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification

[English](README.md) | 简体中文

> S. Chang et al., "A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification", *IEEE Trans. Cogn. Commun. Netw. (2023)*.
> [https://ieeexplore.ieee.org/abstract/document/10239249](https://ieeexplore.ieee.org/abstract/document/10239249)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`fastmldnn`**
（即 `configs/fastmldnn/`）。

## 方法简述

自有方法 A 档：带 I/Q 与 A/P 分支的多损失 FastMLDNN。论文原生划分是 50/10/40 — 不要把残差归因于 TF 的 6:2:2。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/fastmldnn.py::FastMLDNN` |
| Train / test configs | `configs/fastmldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q + A/P |

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
python tools/train.py configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 61.42 / 63.24 | 92.98 / 92.00 | `pass` |
| RML2016.10B | 57.81 / (CSRR-only) | 87.75 / (CSRR-only) | `measured` |
| RML2018.01A | 48.05 / (CSRR-only) | 77.45 / (CSRR-only) | `measured` |
| HisarMod | 5.98 / (CSRR-only) | 7.90 / (CSRR-only) | `measured` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

10A 跟踪通过，61.42/92.98 对论文 63.24/92（近似/接近）。其他数据集仅为实测；Hisar 默认跑崩（约 6%），不能当作复现声明。进一步的论文精确种子/微调围攻已关闭。

