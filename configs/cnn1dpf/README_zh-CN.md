# CNN1DPF — Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks

[English](README.md) | 简体中文

> S. Peng et al. / AMR-Benchmark 1DCNN-PF, "Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks", *ISSCS / related (AMR-Benchmark port)*.
> [https://lirias.kuleuven.be/retrieve/546033](https://lirias.kuleuven.be/retrieve/546033)

CSRR 中的 PyTorch / MMEngine 移植。算法短名 **`cnn1dpf`**
（即 `configs/cnn1dpf/`）。

## 方法简述

并行融合一维 CNN：幅度与相位两路（CSRR 向 A/P 分支送入数据以对齐 TF 的 `to_amp_phase`）分别卷积后再融合分类。

## 论文章节 → 代码对照

| 论文 | 代码 |
|---|---|
| Network / backbone | `csrr/models/backbones/cnn1dpf.py::CNN1DPF` |
| Train / test configs | `configs/cnn1dpf/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P (parallel branches) |

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
python tools/train.py configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py

# 测试一个 checkpoint
python tools/test.py configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## 结果

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.56 / 57.00 | 90.52 / 85.00 | `pass` |
| RML2016.10B | 58.45 / 62.00 | 89.62 / 88.00 | `fail` |
| RML2018.01A | 55.95 / 57.00 | 90.87 / 91.00 | `pass` |
| HisarMod | 42.18 / (CSRR-only) | 65.97 / (CSRR-only) | `measured` |

数字来自官方 `configs/` 根配置的实测结果，对照已发表或常用引用目标（总体准确率 ≥ 目标−2.0 个百分点，峰值 ≥ 目标−1.5 个百分点）。

## 已记录的偏差 / 说明

在 50/10/40 下，10B 总体仍未达到近似门槛。Hisar 仅为 CSRR 实测。TF 同样使用 A/P — 不要改成原始 I/Q。

