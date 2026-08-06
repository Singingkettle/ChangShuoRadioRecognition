# DetectionIsEasy — Detection Is Easy, Recognition Is Hard

Reproduction code for the wideband detection+recognition study:

> S. Chang, Z. Yang, J. He, S. Huang, and Z. Feng, "Detection Is Easy, Recognition Is
> Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition,"
> IEEE Transactions on Cognitive Communications and Networking (TCCN), under review.

Companion locations: the ablation configs live in [`configs/detection_is_easy/`](../../configs/detection_is_easy),
the campaign tools in [`tools/detection_is_easy/`](../../tools/detection_is_easy).

## Method in one paragraph

Wideband spectrum sensing is cast as object detection on an STFT spectrogram. Two findings
drive everything. First, localization is saturated: a vision detector reaches class-agnostic
box mAP ~0.948 — finding the signals is easy. Second, fine-grained recognition is the gap:
57-class class-aware mAP is only ~0.45, because the spectrogram under-uses the phase that
carries modulation identity. The paper ablates the pure-vision recipe along input
representation, phase utility, detector complexity, and detector family, then adds a
domain-matched return-to-IQ branch: boxes labeled as constellation families (PSK/ASK/QAM)
are channelized back to baseband IQ and re-classified by a 1-D hierarchical recognizer,
which lifts deployment mAP by +0.024 — with the recognizer's training budget, not its
architecture, as the decisive lever.

## Paper section → code map

| paper | code |
|---|---|
| Detector ablation grid (input rep / complexity / family) | `configs/detection_is_easy/rtmdet_*`, `fcos_*`, `atss_*`, `yolox_*`, `faster_rcnn_*`, `cascade_rcnn_*`, `deformable_detr_*` |
| STFT / raw-IQ Load transforms, complex data preprocessors, complex-1D backbone | `tools/detection_is_easy/mmdet_plugins.py` |
| Complex-1D primitives + analytic filterbanks | `tools/detection_is_easy/iqdet_complex.py` |
| Return-to-IQ recognizer backbone (1-D ResNet, iq/diff/iqdiff) | `csrr/models/backbones/returniq_resnet1d.py` |
| Hierarchical AMC head (coarse router + 45-class single + 12-class OFDM) | `csrr/models/heads/hierarchical_amc_head.py` |
| Channelized-crop dataset (57-class, `*_L1024.npz` caches) | `csrr/datasets/wideband_channelized.py` |
| Recognizer training recipe (120 ep AdamW + cosine + EMA + label smoothing) | `configs/detection_is_easy/returniq_resnet1d_{iq,diff,iqdiff}_120e_wideband.py` |
| Detect → channelize → recognize bridge, oracle bounds, diagnostics | `tools/detection_is_easy/bridge.py` |
| Class-aware detection mAP + time-frequency IoU metrics | `tools/detection_is_easy/iqdet_metrics.py` |
| Wideband data generation (TorchSig) + COCO export + memmap packing | `tools/detection_is_easy/prepare_torchsig_iq_stratified.py`, `export_*_coco_from_raw.py`, `make_stft_feature_tensor_from_complex.py`, `pack_coco_tensors_to_memmap.py`, `build_multiclass_coco.py` |
| Paper figures + corrected block-SNR analysis | `tools/detection_is_easy/make_figs.py`, `render_example.py`, `analyze_snr_stratified.py`, `analyze_box_quality.py` |

## Data

Synthetic wideband captures from TorchSig (custom 57-class configuration; 50k/5k/10k
train/val/test scenes, 262144 samples each). The heavy generated assets (~191 GB STFT
memmap, ~128 GB raw IQ) are not shipped; the scripts regenerate them:

```bash
cd <repo-root>
python tools/detection_is_easy/prepare_torchsig_iq_stratified.py ...   # raw IQ scenes
python tools/detection_is_easy/export_stft_coco_from_raw.py ...        # STFT tensors + COCO
python tools/detection_is_easy/make_stft_feature_tensor_from_complex.py ...
python tools/detection_is_easy/pack_coco_tensors_to_memmap.py ...      # fast-train memmap
python tools/detection_is_easy/build_multiclass_coco.py ...            # 57-class annotations
```

## Train / evaluate

```bash
# 0) extra requirements on top of a working CSRR environment
pip install -r requirements/detection_is_easy.txt

# 1) detector (the ablation): train a 57-class detector, dump test predictions.
#    Headline metric = the run's coco/bbox_mAP (class-aware, averaged over 57 classes).
python tools/detection_is_easy/run_mmdet_train_eval.py \
  --root <coco_multiclass> \
  --config configs/detection_is_easy/rtmdet_m_stft3_tensor_memmap_resize512.py \
  --epochs 20 --batch-size 8 --optimizer AdamW --lr 5e-4 --seed 0 --dump-results
#    swap --config to sweep the axes:
#      input rep:   rtmdet_m_complex_stft / rtmdet_m_rawiq_fourier_logmag2ch_resize512 (phase out)
#                   rtmdet_m_rawiq_fourier_realimag_resize512 / *_learnable_* / rtmdet_m_complexiq1d_fftbridge_resize512
#      complexity:  rtmdet_{tiny,s,m,l}_stft3_tensor_memmap_resize512
#      family:      fcos_stft3_memmap_resize512 / atss_stft3_memmap_resize512

# 2) recognizer (CSRR-native): cache channelized crops, then train with the CSRR trainer.
python tools/detection_is_easy/bridge.py build --split train --L 1024   # (+ val, test)
python tools/train.py configs/detection_is_easy/returniq_resnet1d_iq_120e_wideband.py

# 3) bridge (deployment): route constellation-family boxes back to raw IQ, re-classify.
python tools/detection_is_easy/bridge.py bridge \
  --baseline-pred <test_predictions.bbox.json> \
  --hier-model recognizer_hier.pth --hier-coarse-route --input-rep iq
#    `oracle` gives the GT-box upper bound; `diag-quality` dumps per-detection diagnostics.

# 4) figures
python tools/detection_is_easy/make_figs.py
python tools/detection_is_easy/render_example.py
```

The 120-epoch AdamW + cosine + EMA + label-smoothing recipe is the lever that lifts
recognition; a short schedule leaves accuracy on the table and can masquerade as a
structural limit.

## Results (3-seed, corrected block-SNR)

| Axis | Result |
|---|---|
| Localization (class-agnostic) | box mAP ~0.948 — saturated |
| Class-aware (RTMDet-M, STFT3) | mAP ~0.447 (seed 7; 0.460±0.011) — the gap |
| Complexity tiny/S/M/L (uniform recipe) | 0.431 / 0.449 / 0.460 / 0.462 — recipe-bound, see paper |
| Phase test (mag-only vs phase+mag) | tie (0.455±0.023 vs 0.447) — phase reaches the net, no gain |
| Learned front end (filterbank) | 0.412 < frozen — do not learn the front end |
| Complex-1D + FFT bridge | collapses (0.026) — FFT of learned features breaks the frequency axis |
| Return-to-IQ deployment | +0.024 overall (0.522 → 0.546); PSK +0.14, ASK +0.12, QAM +0.08 |

## Documented deviations / notes

- **Block-SNR correction.** All SNR-stratified results use
  `block_snr = snr_db + 10*log10(1/(tf*ff))`; do not label results "low-SNR" on the raw axis.
- **mmcv `_ext` stub.** For CPU-only import/smoke runs, the tools call
  `maybe_stub_mmcv_ext()` (in `run_mmdet_smoke.py`) to stub compiled mmcv ops.
- **Synthetic provenance.** Classes, boxes, and SNR are generator ground truth; there is no
  measurement noise floor to hide behind — the recognition gap is structural, and the
  released configs make every number regenerable.

Licensed under the Apache License, Version 2.0.
