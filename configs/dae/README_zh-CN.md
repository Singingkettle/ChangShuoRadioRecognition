# DAE — Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder

[English](README.md) | 简体中文

> S. Rajendran et al., "Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder", *IEEE Trans. Cogn. Commun. Netw. (2021)*.
> [https://ieeexplore.ieee.org/abstract/document/9487492](https://ieeexplore.ieee.org/abstract/document/9487492)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`dae`**
（即 `configs/dae/`）。

## 方法简述

带分类损失与重构损失的 LSTM 自编码器（CSRR `DAEHead`）。输入为幅度/相位，并在幅度通道上做 L2。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/dae.py::DAE` |
| Train / test configs | `configs/dae/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P + reconstruction |

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
python tools/train.py configs/dae/dae_ap-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/dae/dae_ap-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 55.60 / 57.00 | 84.68 / 82.00 | `pass` |
| RML2016.10B | 63.20 / 62.00 | 93.24 / 85.00 | `pass` |
| RML2018.01A | 61.44 / 55.00 | 96.55 / 90.00 | `pass` |
| HisarMod | 54.27 / 40.00 | 61.39 / 70.00 | `fail` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

RML 各集通过。Hisar 峰值失败（论文本身也指出 HisarMod 上混淆严重）。总体可以超过约 40% 的软读数，但峰值仍低于约 70%。

