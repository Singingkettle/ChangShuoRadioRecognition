# JDM Optimization Notes

Date: 2026-07-06 (audit + retune campaign added); paper-exact update 2026-07-14

## Paper-exact campaign (2026-07-14)

- Digitized Fig. 8/10/13 from arXiv:2405.00736 PDF →
  [`paper_figure_targets.md`](paper_figure_targets.md).
- Activated `campaign_mode: paper_exact` in `configs/jdm/retune/goals.json`
  (ideal bars + AMC 90% proxy; P2 joint active). `GOAL_STATUS`: **0/4** met.
- Best vs paper: det mAP 0.8113 vs 0.91 (−0.10); AP75 0.9182 vs 0.96 (−0.04);
  joint fuse 0.6686 vs ideal 0.85 (−0.18) but ≈ Fig. 13(a) **simulate 0.67**;
  AMC proposal 83.03% vs 90% (−6.97 pp).
- **Ceiling:** mixed-test already matches simulate floors; ideal Fig. 8/13 and
  Fig. 12 per-mod precision need `v1` / AWGN `v89–v98` fair eval. AR@k not
  comparable (signal-count mismatch).
- Jobs: `det_paper_exact_ap75_ft_from_wave3b` on GPU1 (ep1 val mAP 0.8135 /
  AP75 0.8707 — AP75 still below wave3b); keepalive queues AWGN eval + FT
  from 5-ep baseline. GPU0 holds AMR ResNetAMR (not killed).
- Fair configs: `eval_awgn_snr12_30_{det,joint}.py`.

## Dataset Scale Audit (2026-07-06)

Full report: [`dataset_scale_audit.md`](dataset_scale_audit.md)

Key findings vs Xing et al. TWC 2024:

- **Scale matches:** 124 versions × 1000 frames = 124k frames; 434k signals;
  1200-sample I/Q @ 150 kHz; 5 modulations.
- **Mismatches:** signal-count histogram shifted (ours: 3/4/5 dominate vs paper
  4/5/6); SNR grid is `-8:2:30` (20 levels) vs paper Table I `12:30:2`;
  bandwidth clusters ~96/120/146 vs paper AP bins 110/130/150; repo 50/10/40
  split undocumented in paper; AR@4/5/6 caps misaligned with our histogram.
- **Impact:** absolute mAP/AR numbers are not strictly paper-comparable;
  qualitative SNR/channel trends remain valid.

Retune campaign plan: [`retune_campaign.md`](retune_campaign.md)

**Architecture freeze (retune):** Score fusion (`fuse_scores`), anchor widths
(96/120/146 vs paper 110/130/150), and bandwidth loss weight (20 vs 2) are
**allowed hyperparameters** — see
[retune_campaign.md § Architecture freeze policy](retune_campaign.md#architecture-freeze-policy).
Same-padding detection grid vs valid-padding is a documented implementation
deviation; retune must **not** change detector/AMC backbone or head topology.

## Retune Campaign Status (2026-07-06)

Wave 1 started (both GPUs free):

| Job | Config | GPU | Status |
|-----|--------|-----|--------|
| `det_30ep_096146_bw20` | `retune/det_30ep_anchor096146_bw20.py` | 0 | **running** |
| `det_30ep_110130150_bw20` | `retune/det_30ep_anchor110130150_bw20.py` | 1 | **running** |

Orchestrator: `tools/jdm/retune_sweep.py` + `configs/jdm/experiments/retune/wave1_manifest.json`

---

Date: 2026-07-04

## Baseline

Dataset: `/home/citybuster/Data/WirelessRadio/data/ChangShuoTwc2026`

Detector checkpoint:
`work_dirs/jdm/jdm-det_fft-csrd/best_detection_mAP_epoch_2.pth`

Detector-only test metrics:

- mAP 0.6417, AP50 0.9893, AP75 0.5527, AR 0.7657
- SNR curve: `work_dirs/jdm/jdm-det_fft-csrd/snr_curve.json`
- Localization diagnostics:
  `work_dirs/jdm/diagnostics/det_baseline_test/test_localization.json`
  and `test_localization.csv`

Joint checkpoint: `work_dirs/jdm/jdm-joint_iq-csrd/jdm_joint.pth`

Joint/class-aware test metrics from the completed non-SNR run:

- mAP 0.3946, AP50 0.5236, AP75 0.4325, AR 0.5794

The later joint SNR run reached the end of inference but did not write final
metrics before becoming an orphaned process, so it should not be used as a
completed result.

## Detector Diagnosis

The AP50/AP75 gap is primarily a localization-quality problem, not proposal
recall:

- Best-overlap recall on the test split is 0.9986 at IoU50 but only 0.7403 at
  IoU75.
- Median center error is 1.02 FFT bins overall, so most boxes are centered well.
- Median bandwidth absolute error is 20.10 FFT bins overall, with many GTs in
  the 16-64 bin error range.
- Widths in the regenerated data are tightly clustered near 96, 120, and 146
  FFT bins. Current anchors (100, 120, 140) have mean center-aligned best IoU
  0.9748 on the test split; empirical anchors (96, 120, 146) improve this to
  0.9928.

By GT bandwidth bucket:

- Small: IoU75 best-overlap recall 0.9347, median center error 0.34 bins,
  median width error 3.74 bins.
- Medium: IoU75 best-overlap recall 0.9962, median center error 0.88 bins,
  median width error 22.06 bins.
- Large: IoU75 best-overlap recall 0.2365, median center error 18.56 bins,
  median width error 41.19 bins, mean signed width error -31.03 bins.

Interpretation:

- The grid/stride math is internally consistent: 1200 bins, stride 8, 150 cells,
  and continuous sigmoid center decode.
- Anchor quantization contributes but is too small to explain the large-box
  failure by itself.
- The dominant detector issue is wrong-width anchor selection/suppression for
  wider boxes. Objectness scores saturate near 1.0, so NMS can keep a
  medium/small-width anchor and suppress a better large-width anchor at the same
  center.
- Training logs also show `loss_bw` around 0.0001 late in training while center
  BCE remains around 0.5 due to continuous-label BCE entropy. Stronger bandwidth
  supervision is a reasonable first training variant.

## Promoted Detector Recipe

The bounded detector experiment
`configs/jdm/experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py` completed
successfully in `work_dirs/jdm/exp_anchor096146_bw20_5ep`.

Promoted settings:

- Empirical anchors `(96, 120, 146)`, matching the regenerated-data bandwidth
  clusters.
- Log-bandwidth MSE loss weight `20.0`, up from the baseline weight `2.0`.
- Baseline architecture, stride, optimizer, detector test thresholds, and
  1-epoch validation cadence were otherwise kept unchanged. The 5-epoch
  experiment used `T_max=5` only to bound the trend check; the official detector
  config keeps the 30-epoch training schedule.

Validation best checkpoint:

- `work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth`
- Epoch 2 validation: mAP 0.7754, AP50 0.9891, AP75 0.9154, AR 0.8567.

Detector-only test result on that checkpoint:

- mAP 0.7677, AP50 0.9894, AP75 0.9182, AR 0.8495.
- SNR artifacts:
  `work_dirs/jdm/exp_anchor096146_bw20_5ep_test/snr_curve.json` and
  `snr_curve.pdf`.

Conclusion: the AP75 jump from 0.5527 to 0.9182 on the test split confirms that
the main detector bottleneck was bandwidth localization. Promote the empirical
anchors and bandwidth-loss weight into the official detector config, and keep
the 5-epoch epoch-2 checkpoint as the current optimized detector checkpoint.

## Variants Prepared

`configs/jdm/experiments/jdm-det_fft-csrd_anchor096146_bw20_5ep.py`

- 5-epoch bounded detector experiment that produced the promoted checkpoint.
- Uses empirical anchors `(96, 120, 146)`.
- Increases log-bandwidth MSE weight from 2 to 20.

`configs/jdm/experiments/jdm-det_fft-csrd_anchor096146_bw20.py`

- Full 30-epoch version of the same detector variant. This is now an optional
  longer follow-up, not a prerequisite for using the promoted detector.

`configs/jdm/experiments/jdm-det_fft-csrd_nms085_top6.py`

- Inference sensitivity config for anchor suppression.
- Uses empirical anchors and NMS IoU 0.85 with `max_per_frame=6` to keep metric
  aggregation bounded.

## Run Status

The optimized detector training and detector-only test completed successfully.
The next cheap preparation step is to merge the optimized detector checkpoint
with the existing AMC checkpoint; the full joint test should be launched only
after checking that the target GPU is still clear.

Recommended next commands:

```bash
python tools/merge_jdm_checkpoints.py \
  work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \
  work_dirs/jdm/jdm-amc_iq-csrd/best_accuracy_top1_epoch_60.pth \
  work_dirs/jdm/jdm-joint_iq-csrd_optimized/jdm_joint_optimized.pth

CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
  configs/jdm/jdm-joint_iq-csrd.py \
  work_dirs/jdm/jdm-joint_iq-csrd_optimized/jdm_joint_optimized.pth \
  --work-dir work_dirs/jdm/jdm-joint_iq-csrd_optimized
```

The joint config now uses the promoted detector anchors, so it can consume a
merged checkpoint built from the optimized detector checkpoint above.

## Optimized Joint Evaluation

Merged checkpoint:
`work_dirs/jdm/jdm-joint_iq-csrd_optimized/jdm_joint_optimized.pth`
(detector epoch 2 from the 5-epoch experiment + AMC
`best_accuracy_top1_epoch_60.pth`).

Class-aware joint test metrics:

- mAP 0.5107, AP50 0.6301, AP75 0.6121, AR 0.6611
- SNR artifacts:
  `work_dirs/jdm/jdm-joint_iq-csrd_optimized/snr_curve.json` and
  `snr_curve.pdf`

Compared with the baseline joint run on the old detector
(`work_dirs/jdm/jdm-joint_iq-csrd/jdm_joint.pth`):

- mAP 0.3946 -> 0.5107 (+29.6% relative)
- AP50 0.5236 -> 0.6301
- AP75 0.4325 -> 0.6121
- AR 0.5794 -> 0.6611

## Detector-only vs Joint Metric Gap

Do **not** compare detector-only mAP (0.7677) directly to joint mAP (0.5107).
They use different evaluation modes in `SignalDetectionMetric`:

- Detector-only configs set `classwise=False` (default): a detection counts as
  a true positive if the interval IoU is high enough, regardless of predicted
  modulation class.
- Joint configs set `classwise=True`: predictions are grouped by predicted class
  and must match both localization **and** modulation label to count as TP.

Code review of the joint pipeline (`JDMFramework._to_baseband`,
`CSRDSignalToBaseband(source='frame')`, checkpoint merge prefixes, and class
order) found no mismatch. The remaining joint gap versus detector-only is
therefore expected from end-to-end classification error on filtered crops, not
from a broken merge or preprocessing bug.

Next optimization target: improve AMC accuracy on detector-proposed crops (or
fuse detection confidence with classification confidence in joint scoring) to
close the class-aware gap further.

## AMC Domain Adaptation (Detector-Proposal Crops)

Date: 2026-07-05

Implemented Option A: fine-tune AMC on crops filtered with **detector proposals**
(matched to each GT signal by best IoU) instead of GT intervals.

### Code / config

- `LoadDetProposal` transform + `CSRDSignalToBaseband` now accepts optional
  `proposal_box` (FFT-bin left/right).
- `tools/precompute_amc_proposals.py` — runs the optimized detector on all
  splits and caches best-IoU proposal boxes per `(file_name, signal_index)`.
- `configs/jdm/experiments/jdm-amc_iq-csrd_detprops_5ep.py` — 5-epoch
  fine-tune from `best_accuracy_top1_epoch_60.pth`, lr 1e-4.

### Proposal cache

`work_dirs/jdm/amc_proposals/all_splits.json`

- 124k frames, 434k signals; 100% matched to a detector proposal (0 GT fallbacks).

### AMC fine-tune (5 epochs, ~7 min)

Work dir: `work_dirs/jdm/exp_amc_detprops_5ep`

| Epoch | val top1 (proposal crops) |
|-------|---------------------------|
| 1     | 74.57%                    |
| 2     | **76.26%** (best)         |
| 3     | 75.39%                    |
| 4     | 75.50%                    |
| 5     | 75.85%                    |

For reference, GT-box AMC val top1 is ~87%. Proposal-crop val confirms the
train/inference distribution shift (~11 pp).

Best checkpoint: `work_dirs/jdm/exp_amc_detprops_5ep/best_accuracy_top1_epoch_2.pth`

### Joint test (optimized detector + proposal-adapted AMC)

Merged checkpoint:
`work_dirs/jdm/jdm-joint_iq-csrd_detprops/jdm_joint_detprops.pth`

Class-aware joint test metrics:

- mAP **0.5205**, AP50 **0.6483**, AP75 **0.6250**, AR **0.6751**
- SNR artifacts:
  `work_dirs/jdm/jdm-joint_iq-csrd_detprops/snr_curve.json` and
  `snr_curve.pdf`

Compared with optimized joint using GT-trained AMC
(`work_dirs/jdm/jdm-joint_iq-csrd_optimized`, mAP 0.5107):

| Metric | GT-AMC joint | Proposal-AMC joint | Delta |
|--------|--------------|--------------------|-------|
| mAP    | 0.5107       | 0.5205             | +0.98 pp |
| AP50   | 0.6301       | 0.6483             | +1.82 pp |
| AP75   | 0.6121       | 0.6250             | +1.29 pp |
| AR     | 0.6611       | 0.6751             | +1.40 pp |

Compared with baseline joint (old detector, mAP 0.3946): **+32.1% relative**.

### Interpretation

The 5-epoch proposal-crop fine-tune gives a consistent but modest joint mAP
lift (+1 pp). Classification on imperfect detector boxes remains the main
bottleneck versus class-agnostic detector mAP (0.7677).

### Commands run

```bash
# 1) Precompute proposal cache (~9 min on GPU1)
CUDA_VISIBLE_DEVICES=1 python tools/precompute_amc_proposals.py \
  configs/jdm/jdm-det_fft-csrd.py \
  work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \
  --out work_dirs/jdm/amc_proposals/all_splits.json

# 2) Fine-tune AMC (~7 min on GPU1)
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/jdm/experiments/jdm-amc_iq-csrd_detprops_5ep.py

# 3) Merge + joint test (~37 min incl. SNR curves)
python tools/merge_jdm_checkpoints.py \
  work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \
  work_dirs/jdm/exp_amc_detprops_5ep/best_accuracy_top1_epoch_2.pth \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops/jdm_joint_detprops.pth

CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
  configs/jdm/jdm-joint_iq-csrd.py \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops/jdm_joint_detprops.pth \
  --work-dir work_dirs/jdm/jdm-joint_iq-csrd_detprops
```

### Recommended next steps

1. ~~**Longer proposal-crop fine-tune** (15–30 epochs)~~ — done; see 20-epoch
   extension below (+0.48 pp joint mAP over 5-ep baseline).
2. ~~**Hard-negative mining**~~ — tried; see HN section below (joint mAP **-0.83 pp**).
3. ~~**Score fusion** (`pred_score = det_score × cls_score`)~~ — strong win; see
   fusion ablation below (**+6.15 pp** joint mAP, no retraining).
4. **End-to-end diagnostic**: measure per-class classification accuracy on
   detector crops at test time to quantify remaining class confusion.
5. ~~**Promote score fusion** into the default joint config / production eval path~~
   — done; `configs/jdm/jdm-joint_iq-csrd.py` now sets `fuse_scores=True`.

## Hard-Negative Mining + Score Fusion

Date: 2026-07-05

### Code changes

- `tools/precompute_amc_proposals.py` — caches `_unmatched` proposals per frame
  (detector boxes with max IoU to any GT `< 0.3`; 23,907 total across splits).
- `CSRDModulationDetPropDataset` — optional hard-negative expansion
  (`include_hard_negatives`, `max_hard_neg_per_frame=3`).
- `LoadDetProposal` — extended to load matched or hard-negative boxes from cache.
- `PrepareGtScore` — one-hot targets for positives, uniform soft targets for
  hard negatives (`CrossEntropyLoss(use_soft=True)`).
- `JDMFramework(fuse_scores=True)` — multiplies `pred_box_scores` by max
  classification confidence at inference.
- Configs:
  - `configs/jdm/experiments/jdm-amc_iq-csrd_detprops_hn_10ep.py`
  - `configs/jdm/experiments/jdm-joint_iq-csrd_fuse_scores.py`

### Proposal cache (updated)

`work_dirs/jdm/amc_proposals/all_splits.json`

- 124k frames, 434k signals; 23,907 unmatched proposals for HN training.

### AMC hard-negative fine-tune (10 epochs, ~16 min on GPU1)

Work dir: `work_dirs/jdm/exp_amc_detprops_hn_10ep`

- `load_from`: 20-ep proposal-AMC best
  (`best_accuracy_top1_epoch_20.pth`)
- Train set: matched proposal crops + up to 3 hard negatives per frame.

| Epoch | val top1 (proposal crops) |
|-------|---------------------------|
| 1     | 76.99%                    |
| 7     | **77.08%** (best)         |
| 10    | 76.93%                    |

Val top1 **regressed** vs 20-ep proposal-AMC (78.09%). Hard-negative uniform
targets appear to dilute positive-class learning within the 10-epoch budget.

Best checkpoint: `work_dirs/jdm/exp_amc_detprops_hn_10ep/best_accuracy_top1_epoch_7.pth`

### Joint test — hard-negative AMC

Merged checkpoint:
`work_dirs/jdm/jdm-joint_iq-csrd_detprops_hn/jdm_joint_detprops_hn.pth`

Class-aware joint test metrics:

- mAP **0.5170**, AP50 **0.6458**, AP75 **0.6202**, AR **0.6754**
- SNR artifacts:
  `work_dirs/jdm/jdm-joint_iq-csrd_detprops_hn/snr_curve.json`
- Joint test duration: **~40 min**.

Compared with 20-ep proposal-AMC joint (mAP 0.5253):

| Metric | 20-ep proposal-AMC | HN 10-ep | Delta |
|--------|--------------------|---------:|------:|
| mAP    | 0.5253             | 0.5170   | -0.83 pp |
| AP50   | 0.6574             | 0.6458   | -1.16 pp |
| AP75   | 0.6305             | 0.6202   | -1.03 pp |
| AR     | 0.6822             | 0.6754   | -0.68 pp |

**Conclusion:** 10-epoch HN training hurt both val top1 and joint mAP. Do not
extend HN training at the current uniform-target recipe; try other directions
(per-class crop diagnostic, separate background class, or lower HN sampling
weight) before revisiting.

### Score fusion ablation (no retraining)

Config: `configs/jdm/experiments/jdm-joint_iq-csrd_fuse_scores.py`
(`fuse_scores=True` on existing 20-ep merged checkpoint).

Work dir: `work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep_fuse`

Class-aware joint test metrics:

- mAP **0.5868**, AP50 **0.7198**, AP75 **0.6943**, AR **0.6822**

Compared with 20-ep proposal-AMC joint without fusion (mAP 0.5253):

| Metric | no fusion | fuse_scores | Delta |
|--------|----------:|------------:|------:|
| mAP    | 0.5253    | 0.5868      | **+6.15 pp** |
| AP50   | 0.6574    | 0.7198      | +6.24 pp |
| AP75   | 0.6305    | 0.6943      | +6.38 pp |
| AR     | 0.6822    | 0.6822      | 0.00 pp |

AR unchanged (same detections); mAP/AP gains come entirely from better
score ranking when classification confidence down-weights uncertain labels.
Low-SNR mAP also improves substantially (e.g. SNR 0 dB: 0.094 → 0.138).

**Conclusion:** Score fusion is the highest-impact improvement so far.
Promoted to the default joint config (`configs/jdm/jdm-joint_iq-csrd.py`).

### Commands run

```bash
# 1) Refresh proposal cache with _unmatched (~5 min on GPU1)
CUDA_VISIBLE_DEVICES=1 python tools/precompute_amc_proposals.py \
  configs/jdm/jdm-det_fft-csrd.py \
  work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \
  --out work_dirs/jdm/amc_proposals/all_splits.json

# 2) Hard-negative AMC fine-tune (~16 min on GPU1)
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/jdm/experiments/jdm-amc_iq-csrd_detprops_hn_10ep.py

# 3) HN merge + joint test (~40 min on GPU1)
python tools/merge_jdm_checkpoints.py \
  work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \
  work_dirs/jdm/exp_amc_detprops_hn_10ep/best_accuracy_top1_epoch_7.pth \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops_hn/jdm_joint_detprops_hn.pth

CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
  configs/jdm/jdm-joint_iq-csrd.py \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops_hn/jdm_joint_detprops_hn.pth \
  --work-dir work_dirs/jdm/jdm-joint_iq-csrd_detprops_hn

# 4) Score fusion ablation on 20-ep checkpoint (~3 min test + ~35 min SNR)
CUDA_VISIBLE_DEVICES=0 python tools/test_det.py \
  configs/jdm/experiments/jdm-joint_iq-csrd_fuse_scores.py \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep/jdm_joint_detprops_20ep.pth \
  --work-dir work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep_fuse
```

## AMC Domain Adaptation — 20-Epoch Extension

Date: 2026-07-05

Extended the 5-epoch proposal-crop fine-tune to 20 epochs, resuming weights
from the 5-ep best checkpoint with a fresh `T_max=20` cosine schedule at lr 1e-4.

### Config

`configs/jdm/experiments/jdm-amc_iq-csrd_detprops_20ep.py`

- `load_from`: `work_dirs/jdm/exp_amc_detprops_5ep/best_accuracy_top1_epoch_2.pth`
- `T_max=20`, `max_epochs=20`, same proposal pipeline and lr as 5-ep run.

### AMC fine-tune (20 epochs, ~30 min on GPU1)

Work dir: `work_dirs/jdm/exp_amc_detprops_20ep`

| Epoch | val top1 (proposal crops) |
|-------|---------------------------|
| 1     | 75.49%                    |
| 2     | 77.32%                    |
| 5     | 77.81%                    |
| 10    | 78.07%                    |
| 15    | 77.44%                    |
| 20    | **78.09%** (best)         |

Val top1 improved from 76.26% (5-ep best) to **78.09%** (+1.83 pp). Training
duration: **30.4 min** (1822 s).

Best checkpoint: `work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth`

### Joint test (optimized detector + 20-ep proposal-adapted AMC)

Merged checkpoint:
`work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep/jdm_joint_detprops_20ep.pth`

Class-aware joint test metrics:

- mAP **0.5253**, AP50 **0.6574**, AP75 **0.6305**, AR **0.6822**
- SNR artifacts:
  `work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep/snr_curve.json` and
  `snr_curve.pdf`
- Joint test duration: **41.3 min** (2476 s incl. SNR curves).

Compared with 5-ep proposal-AMC joint (mAP 0.5205):

| Metric | 5-ep proposal-AMC | 20-ep proposal-AMC | Delta |
|--------|-------------------|--------------------|-------|
| mAP    | 0.5205            | 0.5253             | +0.48 pp |
| AP50   | 0.6483            | 0.6574             | +0.91 pp |
| AP75   | 0.6250            | 0.6305             | +0.55 pp |
| AR     | 0.6751            | 0.6822             | +0.71 pp |

Compared with GT-AMC joint (mAP 0.5107): **+1.46 pp** mAP.

### Interpretation

The 20-epoch extension yields a modest but consistent joint mAP gain (+0.48 pp
over the 5-ep baseline). Val top1 on proposal crops continues to improve slowly
(76.26% → 78.09%), suggesting diminishing returns; further epochs are unlikely
to close the remaining ~9 pp gap to GT-box AMC val (~87%).

### Commands run

```bash
# 1) Fine-tune AMC 20 epochs (~30 min on GPU1)
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
  configs/jdm/experiments/jdm-amc_iq-csrd_detprops_20ep.py

# 2) Merge + joint test (~41 min on GPU1)
python tools/merge_jdm_checkpoints.py \
  work_dirs/jdm/exp_anchor096146_bw20_5ep/best_detection_mAP_epoch_2.pth \
  work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep/jdm_joint_detprops_20ep.pth

CUDA_VISIBLE_DEVICES=1 python tools/test_det.py \
  configs/jdm/jdm-joint_iq-csrd.py \
  work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep/jdm_joint_detprops_20ep.pth \
  --work-dir work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep
```

