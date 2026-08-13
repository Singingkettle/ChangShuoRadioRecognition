# JDM — Joint Signal Detection and Automatic Modulation Classification

> H. Xing, X. Zhang, S. Chang, J. Ren, Z. Zhang, J. Xu, S. Cui,
> "Joint Signal Detection and Automatic Modulation Classification via Deep
> Learning", *IEEE Trans. Wireless Commun.*, vol. 23, no. 11, 2024.
> DOI [10.1109/TWC.2024.3450972](https://doi.org/10.1109/TWC.2024.3450972)
> · arXiv:[2405.00736](https://arxiv.org/abs/2405.00736)

Clean re-implementation on the mmengine `csrr` stack. The reproduction is
**closed** for further numeric siege: detection simulate and AMC match or
exceed the paper; remaining COCO-mAP gaps are high-IoU discretization and
dataset-protocol differences, not a missing model. See
[Results](#results) and [Documented deviations / notes](#documented-deviations--notes).

Companion notes: dataset regen and the once-per-frame noise fix
([`dataset_generation.md`](dataset_generation.md)), digitized Fig. 8/10/13
targets ([`paper_figure_targets.md`](paper_figure_targets.md)).

## Method in one paragraph

A received frame (I/Q, 2×1200 samples at 150 kHz) contains several modulated
signals at different carriers. The **detection module** — a 1-D-conv YOLO-style
network on the FFT (amplitude + phase) — predicts frequency-band proposals
`(center frequency, bandwidth, confidence)`. Because a signal always occupies
the full time axis, boxes are 1-D frequency intervals and IoU/NMS are 1-D.
The **classification module** cuts each proposal out of the frame (carrier
removal + low-pass) into a single-signal baseband crop that a small CNN
classifies into one of five modulations (BPSK / QPSK / 8PSK / 16QAM / 64QAM).
The two modules are trained separately and chained at inference.

## Paper section → code map

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
| Merge separately trained modules | `tools/merge_jdm_checkpoints.py` |
| Detector / joint test entry | `tools/test_det.py` |

## Data

Place the regenerated CSRD / `twc` export at **`data/ChangShuoTwc2026/`**
(symlink is fine). Layout: `v1` … `v124`, each with `anno/*.json` and
`sequence_data/iq/*.mat`. Frames with AWGN use `wideband_data` (noise applied
**once** at the receiver); `signal_data` is noise-free per-signal I/Q.
`CSRDDetectionDataset` / `CSRDModulationDataset` apply a deterministic
50/10/40 train/val/test split per version (seed 0). No split files are stored.

Regenerate with the `twc/` generator in
[ChangShuoRadioData](https://github.com/Singingkettle/ChangShuoRadioData)
(`generate.m` noise policy: one wideband AWGN realization per frame). Protocol
and SNR verification: [`dataset_generation.md`](dataset_generation.md).

```bash
# after generating, point the repo at the export
mkdir -p data
ln -s /path/to/ChangShuoTwc2026 data/ChangShuoTwc2026
```

## Train / evaluate

```bash
# 1) detection module (paper: Adam 1e-3, batch 12, 30 epochs)
python tools/train.py configs/jdm/jdm-det_fft-csrd.py

# 2) classification module (paper: AdamW 1e-3, wd 5e-5, batch 32, 60 epochs)
python tools/train.py configs/jdm/jdm-amc_iq-csrd.py

# 3) stand-alone detection metrics (mixed test; not Fig. 8/13)
python tools/test_det.py configs/jdm/jdm-det_fft-csrd.py \
    work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_*.pth
python tools/test.py configs/jdm/jdm-amc_iq-csrd.py \
    work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_*.pth

# 4) end-to-end joint (detector proposals → classifier)
python tools/merge_jdm_checkpoints.py \
    work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_*.pth \
    work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_*.pth \
    work_dirs/jdm_joint.pth
python tools/test_det.py configs/jdm/jdm-joint_iq-csrd.py work_dirs/jdm_joint.pth

# 5) paper-protocol eval (Fig. 8 / 13). Ideal = v1; simulate = Real + Real_awgn.
python tools/test_det.py \
    configs/jdm/experiments/eval_ideal_v1_det_voted.py \
    work_dirs/jdm/retune/det_full_120ep_lr1e3/best_detection_mAP_epoch_*.pth
python tools/test_det.py \
    configs/jdm/experiments/eval_simulate_real_awgn_det_testonly.py \
    work_dirs/jdm/retune/det_full_120ep_lr1e3/best_detection_mAP_epoch_*.pth
```

Fig. 7 / 10 / 12 SNR curves use AWGN `v89–v98` (`eval_awgn_v89_v98_det_testonly.py`).
Do **not** treat the full 124-version mixed test as Fig. 8/13 simulate.

The operating-point detector is `configs/jdm/experiments/det_full_120ep_lr1e3.py`
(best checkpoint at epoch 4). Ideal joint uses AMC `amc_detprops_120voted_w21`;
simulate joint keeps the W17 fusion (higher AMC top1 on W21 **lowers** simulate
joint mAP).

## Results

Paper Fig. 8/13 numbers are **digitized radar plots** (±0.03 det / ±0.04 joint),
not author tables. Measured values below are test-only on `ChangShuoTwc2026`
(noise applied once per frame). Seed: the promoted det120 / AMC-w21 / AMC-w17
checkpoints; no error bars (single run, same as the paper's unpublished split).

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

Ideal detector per-IoU AP (det120, before voting): ~1.00 through IoU 0.80, then
0.38 / 0.20 / 0.07 at 0.85 / 0.90 / 0.95. Box voting (`vote_iou_thr=0.75`,
`vote_score_pow=4.5`) recovers AP85 and lifts ideal det mAP **0.759 → 0.8254**.
AP90/AP95 stay low (1-D bin / anchor quantization).

## Documented deviations / notes

- **Grid geometry**: same-padding convolutions, three pooling stages → stride-8
  grid of 150 cells for L=1200. Historical code used valid padding and an
  inconsistent feature grid.
- **Anchors**: 3 per cell (paper). Promoted widths **96 / 120 / 146** bins
  (empirical clusters on the regenerated data) with log-bandwidth MSE weight 20.
  Paper text quotes 110 / 130 / 150; historical code used 2 anchors (120 / 90).
- **Low-pass filter**: ideal FFT mask instead of FIR — same brick-wall, shared
  by train crops and inference proposals.
- **Classifier "Sum layer"**: 80-d vector plus `Linear(80, 5)` for logits.
- **Assignment**: YOLOv3-style responsible cell + ignore band (IoU > 0.5).
- **Noise (important)**: original `twc/generate.m` called `awgn` on **every**
  sub-signal, then summed them, so N signals stacked N noise draws (effective
  SNR ≈ label − 10·log10(N)). The May 2024 export gated `awgn` to sub-signal 1
  but dropped fading on the rest for `real` / `real_awgn`. Current generator
  adds noise **once** on the wideband sum (`wideband_data`). **Ideal (v1) never
  had AWGN in any revision**, so the leftover ideal COCO-mAP gap is not an SNR
  bug. Simulate detection already meets Fig. 8 on the *corrected* (harder,
  physically consistent) `real_awgn` data.
- **Signal-count histogram**: paper Fig. 2c is 4/5/6-dominated; this export is
  3/4-dominated with no 6-signal frames. AR@4/5/6 is not comparable.
- **Split**: paper does not publish train/val/test fractions; we use 50/10/40.
- **Why we stop retuning**: longer cosine, extra seeds, bandwidth-loss
  multipliers, EMA/SWA, and AMC-only sieges all sat below det120 (best ckpt
  still epoch 2–4). Further loss tweaks chase AP90/95 on a quantized 1-D grid
  and a digitized 0.91 radar spoke, not a missing method.

`configs/jdm/experiments/` holds the paper-protocol evals and the operating-point
train configs. They are not a second architecture.
