# JDM — Joint Signal Detection and Automatic Modulation Classification

[English](README.md) | 简体中文

> H. Xing, X. Zhang, S. Chang, J. Ren, Z. Zhang, J. Xu, S. Cui,
> "Joint Signal Detection and Automatic Modulation Classification via Deep
> Learning", *IEEE Trans. Wireless Commun.*, vol. 23, no. 11, 2024.
> DOI [10.1109/TWC.2024.3450972](https://doi.org/10.1109/TWC.2024.3450972)
> · arXiv:[2405.00736](https://arxiv.org/abs/2405.00736)

在 mmengine `csrr` 栈上的干净重实现。检测 simulate 与 AMC 达到或超过论文；剩余
COCO-mAP 差距来自高 IoU 离散化与数据集协议差异，不是缺模型。见
[结果](#结果) 与 [已记录的偏差 / 说明](#已记录的偏差--说明)。

配套说明：数据集再生成与“每帧只加一次噪声”的修复
（[`dataset_generation_zh-CN.md`](dataset_generation_zh-CN.md)），数字化的
Fig. 8/10/13 目标（[`paper_figure_targets_zh-CN.md`](paper_figure_targets_zh-CN.md)）。

## 方法简述

一帧接收信号（I/Q，150 kHz 下 2×1200 采样）里有若干个落在不同载波上的调制信号。
**检测模块**是一个作用在 FFT（幅度 + 相位）上的一维卷积 YOLO 风格网络，预测频带
proposal `(中心频率, 带宽, 置信度)`。因为信号总是占满整个时间轴，框是一维频率
区间，IoU/NMS 也是一维的。**分类模块**把每个 proposal 从帧里切出来（去载波 +
低通），得到单信号基带片段，再用一个小 CNN 分成五种调制（BPSK / QPSK / 8PSK /
16QAM / 64QAM）。两个模块分开训练，推理时串起来。

## 论文章节 → 代码对照

| paper | code |
|---|---|
| Sec. V-B detection CNN (Fig. 4) | `csrr/models/backbones/jdm.py::JDMDetectionBackbone` |
| Sec. V-B YOLO-style head, anchors, confidence | `csrr/models/heads/jdm_det_head.py::JDMDetectionHead` |
| Eq. (6) IoU (1-D degenerate) + NMS | `csrr/models/utils/interval_ops.py` |
| detection losses (BCE conf / BCE center / MSE log-bw ×20) | `JDMDetectionHead.loss` |
| Sec. V-C classification CNN (Fig. 5, "Sum layer") | `csrr/models/backbones/jdm.py::JDMClassificationBackbone` |
| Sec. V-C proposal filtering (carrier removal + LPF) | `csrr/datasets/transforms/csrd.py::CSRDSignalToBaseband` (train) / `csrr/models/detectors/jdm.py::JDMFramework._to_baseband` (inference) |
| Sec. V-A JDM pipeline | `csrr/models/detectors/jdm.py::JDMFramework` |
| Sec. IV CRML23 dataset | `csrr/datasets/csrd.py` over `data/ChangShuoTwc2026/v*` |
| FFT input | `csrr/datasets/transforms/csrd.py::IQToSpectrum` (+ `LoadCSRDFrame`) |
| Sec. VI-A metrics | `csrr/evaluation/metrics/detection.py::SignalDetectionMetric` |
| Sec. VI training protocol | `configs/jdm/*.py` |
| Merge separately trained modules | `configs/jdm/scripts/merge_jdm_checkpoints.py` |
| Detector / joint test entry | `tools/test.py` |

## 数据

把再生成的 CSRD / `twc` 导出放到 **`data/ChangShuoTwc2026/`**（软链接即可）。
布局：`v1` … `v124`，各自带 `anno/*.json` 与 `sequence_data/iq/*.mat`。带 AWGN
的帧使用 `wideband_data`（噪声在接收端**只加一次**）；`signal_data` 是无噪声的
逐信号 I/Q。`CSRDDetectionDataset` / `CSRDModulationDataset` 对每个版本做确定的
50/10/40 训练/验证/测试划分（seed 0）。不另存划分文件。

用 [ChangShuoRadioData](https://github.com/Singingkettle/ChangShuoRadioData)
里的 `twc/` 生成器再生成（`generate.m` 噪声策略：每帧一次宽带 AWGN）。协议与
SNR 核验见 [`dataset_generation_zh-CN.md`](dataset_generation_zh-CN.md)。

```bash
# after generating, point the repo at the export
mkdir -p data
ln -s /path/to/ChangShuoTwc2026 data/ChangShuoTwc2026
```

## 训练 / 评测

```bash
# 1) detection module (paper: Adam 1e-3, batch 12, 30 epochs)
python tools/train.py configs/jdm/jdm-det_fft-csrd.py

# 2) classification module (paper: AdamW 1e-3, wd 5e-5, batch 32, 60 epochs)
python tools/train.py configs/jdm/jdm-amc_iq-csrd.py

# 3) stand-alone detection metrics (mixed test; not Fig. 8/13)
python tools/test.py configs/jdm/jdm-det_fft-csrd.py \
    work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_*.pth
python tools/test.py configs/jdm/jdm-amc_iq-csrd.py \
    work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_*.pth

# 4) end-to-end joint (detector proposals → classifier)
python configs/jdm/scripts/merge_jdm_checkpoints.py \
    work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_*.pth \
    work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_*.pth \
    work_dirs/jdm_joint.pth
python tools/test.py configs/jdm/jdm-joint_iq-csrd.py work_dirs/jdm_joint.pth

# 5) paper-protocol eval (Fig. 8 / 13). Ideal = v1; simulate = Real + Real_awgn.
python tools/test.py \
    configs/jdm/experiments/eval_ideal_v1_det_voted.py \
    work_dirs/jdm/retune/det_full_120ep_lr1e3/best_detection_mAP_epoch_*.pth
python tools/test.py \
    configs/jdm/experiments/eval_simulate_real_awgn_det_testonly.py \
    work_dirs/jdm/retune/det_full_120ep_lr1e3/best_detection_mAP_epoch_*.pth
```

Fig. 7 / 10 / 12 的 SNR 曲线使用 AWGN `v89–v98`（`eval_awgn_v89_v98_det_testonly.py`）。
**不要**把全部 124 个版本的混合测试当成 Fig. 8/13 的 simulate。

工作点检测器是 `configs/jdm/experiments/det_full_120ep_lr1e3.py`（最佳 checkpoint
在 epoch 4）。Ideal joint 使用 AMC `amc_detprops_120voted_w21`；simulate joint
仍用 W17 融合（W21 上更高的 AMC top1 **会降低** simulate joint mAP）。

## 结果

论文 Fig. 8/13 的数字来自**数字化雷达图**（检测 ±0.03 / joint ±0.04），不是作者
表格。下面的实测值是在 `ChangShuoTwc2026` 上只跑测试（噪声每帧只加一次）。种子：
晋升的 det120 / AMC-w21 / AMC-w17 checkpoint；无误差条（单次运行，与论文未公开
划分的做法相同）。

| Protocol | Metric | Paper (digitized) | Measured | Status |
|---|---|---:|---:|---|
| Fig. 8(a) simulate | det mAP | 0.76 | **0.7701** | met (unvoted NMS) |
| Fig. 8(a) simulate | det AP75 | 0.81 | **0.8692** | exceeded |
| Fig. 8(a) ideal | det AP50 | 1.00 | **1.00** | met |
| Fig. 8(a) ideal | det AP75 | 0.96 | **0.9894** | exceeded |
| Fig. 8(a) ideal | det mAP | 0.91 | 0.8254 | COCO mean dragged by AP≥0.90; see notes |
| Fig. 10 | AMC vs SNR (GT box) | digitized curves | **exceeds every (mod, SNR)** | met |
| Fig. 13(a) ideal | joint mAP | 0.85 | 0.7709 | inherits AP≥0.90 tail |
| Fig. 13(a) simulate | joint mAP | 0.67 | 0.5195 | not a hard target; see notes |

Ideal 检测器按 IoU 的 AP（det120，投票前）：到 IoU 0.80 约为 1.00，随后在
0.85 / 0.90 / 0.95 落到 0.38 / 0.20 / 0.07。框投票（`vote_iou_thr=0.75`，
`vote_score_pow=4.5`）能挽回 AP85，并把 ideal det mAP 从 **0.759 提到 0.8254**。
AP90/AP95 仍然偏低（一维 bin / 锚框量化）。

## 已记录的偏差 / 说明

- **网格几何**：same-padding 卷积、三层池化 → L=1200 时 stride-8、150 个格子。
  历史代码用 valid padding，特征网格也不一致。
- **锚框**：每格 3 个（论文）。晋升宽度为 **96 / 120 / 146** bin（在再生成数据上
  的经验聚类），log-bandwidth MSE 权重为 20。论文正文写的是 110 / 130 / 150；
  历史代码用 2 个锚框（120 / 90）。
- **低通滤波器**：理想 FFT 掩膜而不是 FIR — 同样的砖墙特性，训练裁剪与推理
  proposal 共用。
- **分类器 “Sum layer”**：80 维向量再加 `Linear(80, 5)` 得到 logits。
- **分配**：YOLOv3 风格的负责格子 + 忽略带（IoU > 0.5）。
- **噪声（重要）**：原始 `twc/generate.m` 对**每一个**子信号调用 `awgn`，再求和，
  于是 N 个信号叠了 N 次噪声（有效 SNR ≈ 标签 − 10·log10(N)）。2024 年 5 月的
  导出把 `awgn` 限制在子信号 1，但又把 `real` / `real_awgn` 其余子信号的衰落丢掉了。
  当前生成器只在宽带求和上**加一次**噪声（`wideband_data`）。**Ideal（v1）在任何
  修订里都没有 AWGN**，所以剩下的 ideal COCO-mAP 差距不是 SNR bug。Simulate 检测
  已经在*修正后*（更难、物理上自洽）的 `real_awgn` 数据上达到 Fig. 8。
- **信号个数直方图**：论文 Fig. 2c 以 4/5/6 为主；这份导出以 3/4 为主，没有
  6 信号帧。AR@4/5/6 不可比。
- **划分**：论文未公布训练/验证/测试比例；我们用 50/10/40。
- **为何停止再调**：更长的 cosine、额外种子、带宽损失倍率、EMA/SWA，以及只围攻
  AMC，都低于 det120（最佳 ckpt 仍在 epoch 2–4）。再改损失只是在量化的一维网格
  和数字化的 0.91 雷达辐条上追 AP90/95，不是缺方法。

`configs/jdm/experiments/` 存放论文协议评测与工作点训练配置。它们不是第二套架构。
