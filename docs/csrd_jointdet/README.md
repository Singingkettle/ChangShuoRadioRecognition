# JDM — Joint Signal Detection and Automatic Modulation Classification

Clean re-implementation of

> H. Xing, X. Zhang, S. Chang, J. Ren, Z. Zhang, J. Xu, S. Cui,
> "Joint Signal Detection and Automatic Modulation Classification via Deep
> Learning", *IEEE Trans. Wireless Commun.*, vol. 23, no. 11, 2024.
> DOI 10.1109/TWC.2024.3450972 · arXiv:2405.00736

on the current mmengine-based `csrr` stack. Background research and the
inventory of the historical (pre-mmengine) implementation live in
[`paper_and_history_notes.md`](paper_and_history_notes.md).

## Method in one paragraph

A received frame (I/Q, 2×1200 samples at 150 kHz) contains several modulated
signals at different carriers. The **detection module** — a 1-D-conv YOLO-style
network operating on the FFT (amplitude + phase) of the frame — predicts
frequency-band *proposals* `(center frequency, bandwidth, confidence)`.
Because a signal always spans the full time axis, boxes are 1-D frequency
intervals and IoU/NMS are 1-D interval operations. The **classification
module** cuts each proposal out of the frame (carrier removal + low-pass),
producing a single-signal baseband crop that a small CNN classifies into one
of 5 modulations (BPSK/QPSK/8PSK/16QAM/64QAM). The two modules are trained
separately and chained at inference.

## Paper section → code map

| paper | code |
|---|---|
| Sec. V-B detection CNN (Fig. 4) | `csrr/models/backbones/jdm.py::JDMDetectionBackbone` |
| Sec. V-B YOLO-style head, anchors, confidence | `csrr/models/heads/jdm_det_head.py::JDMDetectionHead` |
| Eq. (6) IoU (1-D degenerate) + NMS | `csrr/models/utils/interval_ops.py` |
| detection losses (BCE conf / BCE center / MSE log-bw ×2) | `JDMDetectionHead.loss` (+ `csrr/models/losses`) |
| Sec. V-C classification CNN (Fig. 5, "Sum layer") | `csrr/models/backbones/jdm.py::JDMClassificationBackbone` |
| Sec. V-C proposal filtering (carrier removal + LPF) | `csrr/datasets/transforms/csrd.py::CSRDSignalToBaseband` (train) / `csrr/models/detectors/jdm.py::JDMFramework._to_baseband` (inference) |
| Sec. V-A JDM pipeline ("proposal" hand-off) | `csrr/models/detectors/jdm.py::JDMFramework` |
| Sec. IV CRML23 dataset | `csrr/datasets/csrd.py` (`CSRDDetectionDataset`, `CSRDModulationDataset`) over `data/ChangShuo/v*` |
| FFT input representation | `csrr/datasets/transforms/csrd.py::IQToSpectrum` (+ `LoadCSRDFrame`) |
| Sec. VI-A metrics (mAP/AP50/AP75, size-binned AP, AR@k) | `csrr/evaluation/metrics/detection.py::SignalDetectionMetric` |
| Sec. VI training protocol | `configs/jdm/*.py` (optimizers/epochs/batch sizes per paper) |

Model wrapper for the stand-alone detector: 
`csrr/models/detectors/signal_detector.py::SignalDetector` (mmengine
`BaseModel` with the same loss/predict contract as `SignalClassifier`).

## Data

Expected at `data/ChangShuo/v1 … v124` (CSRD / ChangShuoRadioData `twc`
profile output; present on this machine). Each version = one channel
configuration × 1000 entries: `anno/*.json` + `sequence_data/iq/*.mat`
(`signal_data`: per-signal passband I/Q `(num_signals, 2, 1200)`, frame = sum).
No split files are used; the datasets split every version 50/10/40
(train/validation/test) with a fixed seed.

## Train / evaluate

```bash
# 1) detection module (paper: Adam 1e-3, batch 12, 30 epochs)
python tools/train.py configs/jdm/jdm-det_fft-csrd.py

# 2) classification module (paper: AdamW 1e-3, wd 5e-5, batch 32, 60 epochs)
python tools/train.py configs/jdm/jdm-amc_iq-csrd.py

# 3) stand-alone detection metrics on the test split
python tools/test_det.py configs/jdm/jdm-det_fft-csrd.py \
    work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_*.pth

# classifier accuracy uses the standard classification test entry
python tools/test.py configs/jdm/jdm-amc_iq-csrd.py \
    work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_*.pth

# 4) end-to-end JDM (detector proposals -> classifier), class-aware mAP
python tools/merge_jdm_checkpoints.py \
    work_dirs/jdm-det_fft-csrd/best_detection_mAP_epoch_*.pth \
    work_dirs/jdm-amc_iq-csrd/best_accuracy_top1_epoch_*.pth \
    work_dirs/jdm_joint.pth
python tools/test_det.py configs/jdm/jdm-joint_iq-csrd.py work_dirs/jdm_joint.pth
```

To restrict training/evaluation to specific channel conditions (e.g. only the
AWGN versions, mirroring the paper's per-condition figures), override
`versions`, e.g. `--cfg-options test_dataloader.dataset.versions=v1`.

## Documented deviations from the paper

- **Grid geometry**: same-padding convolutions with three pooling stages give
  an exact stride-8 grid of 150 cells for L = 1200. The paper/historical code
  used valid padding, which produced a feature grid inconsistent with the
  anchor stride (see the history notes).
- **Anchors**: 3 per cell (paper) with widths 100/120/140 bins, chosen from
  the dataset's bandwidth clusters; the historical code used 2 anchors
  (120/90).
- **Low-pass filter**: ideal (FFT-mask) filtering instead of a FIR filter —
  same brick-wall behaviour, simpler and identical between training crops and
  inference proposals.
- **Classifier output**: the paper's "Sum layer" yields an 80-dim vector; a
  final `Linear(80, 5)` projection (implicit in the paper) produces the class
  logits.
- **Assignment details**: YOLOv3-style responsible-cell assignment with an
  ignore band (IoU > 0.5) instead of mmdet's `GridAssigner(pos=neg=0.95)`
  hack in the historical code.
