# Reproduction record

English | [简体中文](REPRODUCTION_zh-CN.md)

This file records an end-to-end re-run of the paper's experiments from a fresh clone of the
released code, and states, cell by cell, where the re-run lands relative to the reported
numbers. It is a factual log. It draws no conclusions the measurements do not force.

Every number below was produced on server hardware from the released `main` branch. The
comparison table is generated mechanically by `configs/detection_is_easy/collect_repro_results.py`
against `configs/detection_is_easy/paper_values.csv`, which stores each reported value next to
the table or section it came from.

**Read the `paper` columns with one fact in mind.** The manuscript has since been revised
against this re-run: the paper values in `paper_values.csv`, and the `paper (current)` column
in every table below, are the manuscript's *current* numbers, and they were corrected on the
basis of this very reproduction. Agreement between `reproduced` and `paper (current)` is
therefore not an independent check -- it is circular by construction. For that reason every
table also carries an `originally reported` column, the value in the manuscript as first
submitted, and the section "What was corrected, and why" records each change and its cause.
The `reproduced` columns are untouched measurements.

## What was run, on what

- **Code**: a clean `git clone` of this repository at commit `88c02ff` (the two data-path
  fixes described below are in it). The reporting tool `collect_repro_results.py` was added
  afterwards at `c997b1c`; it reads results and changes none.
- **Environment**: a fresh venv built strictly from `requirements/detection_is_easy.txt` --
  torch 2.7.1+cu128, numpy 2.2.6, mmdet 3.3.0, **mmcv-lite 2.1.0** (no compiled `_ext`),
  mmengine 0.10.7, torchsig 2.1.1. Verified package-for-package against the environment the
  paper ran on.
- **Hardware**: 8x RTX 4090.
- **Data**: the released `hardshort_lowsnr` benchmark -- 50000/5000/10000 train/val/test,
  57 classes -- with the 57-class annotations rebuilt on the machine by
  `build_multiclass_coco.py` (a match rate of 1.0000, identical box counts to the original).

## How a cell counts as reproduced

`cfg.randomness = dict(seed=..., deterministic=False)`, so cuDNN picks non-deterministic
kernels and two runs of one seed differ. We measured that floor before judging anything:
three identical runs of RTMDet-M gave sd **0.0076**, of RTMDet-tiny sd **0.0033**. A cell
counts as reproduced when its seed mean sits within **0.023** (three times the larger floor)
of the reported value. This is a band, not a point; a single pair of runs cannot carry a
0.01 difference on this benchmark.

Two run-time contracts matter and are recorded per run in `run_info.json`:

- **`used_mmcv_lite_stub: true`** -- every result uses the pure-PyTorch NMS fallback, because
  the paper's environment has no compiled mmcv. Installing full CUDA mmcv swaps the NMS
  implementation and shifts numbers slightly.
- **`used_pytorch_focal_loss: true`** -- FCOS and ATSS need a focal-loss kernel mmcv-lite does
  not ship; the harness routes them to mmdet's own `py_sigmoid_focal_loss` (same quantity).
  See "Reproducibility defects fixed" below.

## Detector cells: measured vs reported

Class-aware `coco/bbox_mAP` over the 57 categories, validation split. `n` is the seed count;
`sd` is the spread across seeds. `paper (current)` is the manuscript's current value (mean
+- sd where it is a three-seed cell); `originally reported` is the value in the manuscript
as first submitted, with its seed count where that differed; `delta` and "verdict" are
`reproduced` minus `paper (current)`, against the 0.023 band.

| Cell | reproduced | sd | paper (current) | originally reported | delta vs current | verdict | paper source |
|---|---|---|---|---|---|---|---|
| Axis B, tiny, uniform | 0.432 | 0.0035 | 0.432 +- 0.004 | 0.431 | 0.000 | within | Table I uniform |
| Axis B, small, uniform | 0.443 | 0.0205 | 0.443 +- 0.021 | 0.449 | 0.000 | within | Table I uniform |
| Axis B, medium, uniform | 0.472 | 0.0121 | 0.472 +- 0.012 | 0.460 | 0.000 | within | Table I uniform |
| Axis B, large, uniform | 0.451 | 0.0074 | 0.451 +- 0.007 | 0.462 | 0.000 | within | Table I uniform |
| Axis B, tiny, own sched. | 0.433 | 0.0035 | 0.433 +- 0.004 | 0.408 (2 seeds) | 0.000 | within | Table I own |
| Axis B, small, own sched. | 0.470 | 0.0135 | 0.470 +- 0.014 | 0.429 | 0.000 | within | Table I own |
| Axis B, medium, own sched. | 0.486 | 0.0152 | 0.492 +- 0.010 (batch 4) | 0.477 +- 0.039 (pooled with the batch-8 run) | -0.006 | within | Table I own |
| Axis B, large, own sched. | 0.504 | 0.0154 | 0.504 +- 0.015 | 0.486 (2 seeds) | 0.000 | within | Table I own |
| Axis A, STFT3 offline (reference) | 0.472 | 0.0121 | 0.472 +- 0.012 | -- (not in the first submission) | 0.000 | within | Table III; same runs as Axis B medium/uniform by construction |
| Axis A, magnitude-only | 0.441 | 0.0231 | 0.441 +- 0.023 | 0.455 | 0.000 | within | Table III |
| Axis A, phase+magnitude | 0.455 | 0.0125 | 0.455 +- 0.013 | 0.447 (2 seeds) | 0.000 | within | Table III |
| Axis A, phase only | 0.431 | -- | 0.431 | 0.440 | 0.000 | within | Table III |
| Axis A, learnable filterbank | 0.418 | -- | 0.418 | 0.412 | 0.000 | within | Table III |
| Axis E, complex-1D + FFT | 0.053 | -- | 0.053 | 0.026 | 0.000 | within | Table III |
| FCOS | 0.470 | 0.0053 | 0.470 +- 0.005 | 0.374 (identity normalisation, 1 seed) | 0.000 | within | Table II (Axis C), §VI-B |
| ATSS | 0.468 | 0.0032 | 0.468 +- 0.003 | 0.380 (identity normalisation, 1 seed) | 0.000 | within | Table II (Axis C), §VI-B |
| RTMDet-M, uniform, batch 4 | 0.426 | 0.0006 | 0.426 +- 0.001 | -- (not in the first submission) | 0.000 | within | Table II (Axis C), §VI-B, matched-batch control |
| localization (single-class) | 0.893 | 0.0060 | 0.893 +- 0.006 | 0.948 (easier generator configuration) | 0.000 | within | Sections I, IV, VI-A |

The `deploy_m_own` cell -- the deployment detector's single best run, originally reported as
0.521 -- is no longer in this table because the manuscript no longer cites that number; the
reproduction of that single run (0.472) and where the number went are recorded under "What
was corrected, and why".

Controls with no paper value of their own:

| Cell | reproduced | sd | why it was run |
|---|---|---|---|
| RTMDet-M, own sched., batch 4 | 0.491 | -- | batch-decoupled size control (§VI-C, L5); the manuscript's current medium/own value (0.492 +- 0.010) is this batch-4 configuration at three seeds |

All 18 cells that carry a current paper value land inside the band. Given the disclosure at
the top of this file, that is what the correction guarantees, not independent evidence for
it. Against the *originally reported* values, seven cells did not land inside the band --
tiny/own, small/own, complex-1D + FFT, FCOS, ATSS, localization, and the deployment detector's
single best run -- and every one of them has since been corrected in the manuscript. Each is
accounted for, with the value it replaced, in "What was corrected, and why" below.

## Recognizer and deployment cells

Recognizer accuracy on the val crop cache (clean signals), and the deployment bridge on the
2963 test scenes with archived raw IQ, off the reproduced deployment detector.

| Cell | metric | reproduced | paper (current) | originally reported | note |
|---|---|---|---|---|---|
| recipe-A recognizer (3 seeds) | combined fine acc | 0.875 / 0.876 / 0.871 | 0.875 | 0.869 | run-index 19 |
| 40-epoch predecessor | stage2-single acc | 0.632 | 0.632 | 0.643 (labelled as combined) | run-index 20 -- see dose-response note |
| 40-epoch predecessor | combined fine acc | 0.534 | 0.534 | -- (not reported) | |
| recipe-B (mixup) | combined fine acc | 0.699 | 0.699 | 0.714 | run-index 21 |
| differential phase | combined fine acc | 0.913 | 0.913 | 0.916 | run-index 26 |
| recipe-A deployment (3 seeds) | fused delta | +0.028 (0.027/0.029/0.027) | +0.028 | +0.024 | run-index 22 |
| recipe-A deployment | psk / ask / qam delta | +0.153 / +0.132 / +0.081 | +0.153 / +0.132 / +0.081 | +0.143 / +0.118 / +0.084 | run-index 23 |
| oracle (perfect box), recipe-A | pure-IQ class mAP | 0.608 | 0.608 | 0.608 | run-index 24 |
| differential phase deployment | fused delta | +0.022 | +0.022 | +0.019 | run-index 26 |
| predicted-box recogniser, RTMDet (3 seeds) | fused delta | +0.092 +- 0.001 | +0.092 +- 0.001 | -0.019 (high-score cache) | see "originally reported as -0.019" below |
| predicted-box recogniser, FCOS (3 seeds) | fused delta | +0.189 +- 0.002 | +0.189 +- 0.002 | -- (not in the first submission) | |
| predicted-box recogniser, RTMDet, per family (3 seeds) | psk / qam / ask delta | +0.453 +- 0.011 / +0.537 +- 0.023 / +0.373 +- 0.016 | +0.453 / +0.537 / +0.373 | psk +0.457 (seed 101 only) | `work_dirs/repro/deployment/bridge_predhi_s{101,202,303}/summary.csv` |

The same caveat as above applies to the `paper (current)` column here: these are the
manuscript's numbers after the correction, and they were set from this re-run.

The deployment bridge uses the recorded non-default flags -- `--score-thr 0.05 --limit 2963
--class-nms-iou 0.5 --ours-score-recog --iq-families psk,ask,qam`. Running with bridge
defaults evaluates a different scene set with per-class NMS off and does not reproduce these.

## What was corrected, and why

Each entry names the value as originally reported, what the re-run measured, and what the
manuscript now says. The deltas quoted in the headings are `reproduced` minus `originally
reported`, i.e. the discrepancy that triggered the correction.

**FCOS +0.096, ATSS +0.088 (§VI-B) -- originally reported as 0.374 / 0.380, corrected to
0.470 +- 0.005 / 0.468 +- 0.003.** The originally reported FCOS/ATSS runs had trained with
identity normalisation -- their generated config carried `mean=[0,0,0] std=[1,1,1]` on data
whose per-channel sigma is about 12.8 -- while the RTMDet runs they were compared against used
the real statistics. Re-running FCOS/ATSS with the statistics injected
(`--require-tensor-stats`) gave 0.470 and 0.468, at three seeds with sd 0.005 and 0.003. Two
further facts sharpened the picture:

- RTMDet-M under the same uniform recipe is 0.472 at batch 8 but **0.426 at batch 4**, the
  batch FCOS/ATSS ran at. At matched batch, FCOS (0.470) and ATSS (0.468) sit *above*
  RTMDet-M (0.426), not below it.
- So the originally reported RTMDet lead came from two compounding artefacts: a normalisation
  defect handicapping the other heads, and a batch-size difference. With both removed, the
  detector family does not separate.

The manuscript adopted this: the identity-normalisation root cause is now stated in §VI-C,
the family comparison reports FCOS/ATSS above RTMDet-M at matched batch (with the batch-4
RTMDet-M control, 0.426 +- 0.001, added as a cell), and the first-round 0.374 / 0.380 survive
only as the history recorded here. The finding is consistent with the paper's thesis --
localization is saturated, the gap is recognition.

**localization -0.055 (§I/IV/VI-A) -- originally reported as 0.948, corrected to
0.893 +- 0.006.** The originally reported class-agnostic localization mAP of 0.948 had been
measured on an easier generator configuration (the `lowsnr` set: longer durations, higher
SNR, no co-channel overlap) than the released `hardshort_lowsnr` benchmark. Three
single-class runs on the released benchmark gave **0.893 +- 0.006**. Localization is still
the easy, near-saturated axis; the manuscript now states it with the number measured on the
benchmark it distributes.

**deployment detector -0.049 -- originally reported as 0.521 (single best run); no longer
cited.** The first submission quoted this cell's single best run (0.521, a batch-8 run) as
the deployment baseline but its three-seed mean (0.477, pooling that batch-8 run with two
batch-4 runs) for the size sweep. The reproduction gave 0.472 for the single run and 0.486
for the pooled three-seed mean. The manuscript no longer cites 0.521: the deployment detector
is stated as the medium/own cell at batch 4 over three seeds, 0.492 +- 0.010 (the reproduced
batch-4 control is 0.491), and the `deploy_m_own` cell was dropped from the table above. The
deployment numbers are all bridged off the deployment checkpoint, so their absolute level
tracks it (reproduced baseline 0.474 against the originally reported 0.522); the *deltas*
reproduced regardless and were refreshed as listed in the recognizer table.

**small/own +0.041, tiny/own +0.025 -- own-schedule column originally reported as
0.408 / 0.429 / 0.477 / 0.486, corrected to 0.433 / 0.470 / 0.492 / 0.504.** Real deviations
in the own-schedule column, whose originally reported spread was itself large (sd 0.017 for
small at three seeds, 0.041 for tiny at two) and whose tiny and large cells rested on two
seeds. The manuscript now reports the whole column at lr 1e-4, batch 4, three seeds:
0.433 +- 0.004 / 0.470 +- 0.014 / 0.492 +- 0.010 / 0.504 +- 0.015 for tiny / small / medium /
large. The uniform column of the same sizes had reproduced cleanly and was refreshed to the
re-run means as well (0.431 / 0.449 / 0.460 / 0.462 -> 0.432 / 0.443 / 0.472 / 0.451).

**complex-1D +0.027 -- originally reported as 0.026, corrected to 0.053.** Both values are a
collapse; the difference has no meaning at that magnitude. The Axis E finding -- a
learned-then-FFT front end destroys localization -- reproduced, and the manuscript now
carries the re-run value.

**Axis A cells that were inside the band but were refreshed.** Magnitude-only 0.455 ->
0.441 +- 0.023, phase + magnitude 0.447 (two seeds) -> 0.455 +- 0.013 (three seeds), phase
only 0.440 -> 0.431, learnable filterbank 0.412 -> 0.418. None of these changed a verdict:
magnitude-only and phase + magnitude still tie within the seed spread, and the learnable
front end still loses to the frozen one. The manuscript also added an offline-STFT3 reference
cell (0.472) to Table III, which the first submission did not list.

**Predicted-box recogniser (§VII) -- originally reported as -0.019, corrected to +0.092.**
The first submission's negative-table row for "train the recogniser on predicted boxes"
reported a deployment delta of -0.019. The reproduction gave **+0.092** (mean of the three per-seed differences, +0.0924; the earlier text printed +0.093, which is the difference of the rounded absolute values 0.567 - 0.474), at three seeds
(+0.0930/+0.0922/+0.0919, sd < 0.001), against the recipe-A baseline of +0.028. Three checks
pinned the mechanism:

- *Not a leak.* Train and test scenes are disjoint (50000 vs 10000, zero shared sample ids or
  file names).
- *Distribution match, not a better recogniser.* On perfect GT boxes this recogniser scores
  0.415, *below* recipe-A's 0.608, and its clean val accuracy is 0.525 vs recipe-A's 0.875. It
  is worse everywhere except on the predicted-box distribution it was trained for.
- *Explains the originally reported -0.019.* The first submission's crop cache was named
  `trainpred_hi`; its build parameters were never logged, and it holds 194k crops. Building
  the analogous cache at a high detection-score cut (`--score-thr 0.5`) keeps 87k
  near-perfect boxes and yields a deployment delta of **-0.027** -- reproducing the
  originally reported -0.019. Training on the full routed-box distribution
  (`--score-thr 0.1`) is what produces the +0.092. The originally published negative result
  was an artefact of building the cache from high-score, near-GT boxes. The manuscript now
  reports the predicted-box recogniser as a positive result at +0.092 +- 0.001 and gives the
  high-score cache as the explanation of the earlier sign.

The parameters of `trainpred_hi` are unrecorded; the +0.092 cache was built at the released
default `--score-thr 0.1` over all 50000 scenes, then randomly subsampled (fixed seed) to
match the GT cache's crop count so the comparison is not handed extra data.

The recipe is per-detector, and it generalizes. Repeating the whole procedure on FCOS --
dump FCOS's train predictions, build the predicted-box cache, train the recogniser on it,
deploy on FCOS's test predictions -- gives **+0.189 +- 0.002** over three seeds
(0.469 -> 0.659), larger than RTMDet's, as the weaker vision head leaves more for the IQ
branch to rescue. The training must match the detector: a recogniser trained on RTMDet's
boxes and deployed on FCOS's, with no retraining, gains only +0.040 (above the GT-box
recipe-A's +0.034 on FCOS, so the distribution transfers in part), and only training on the
detector's own boxes recovers the full gain.

| Deployment detector | recogniser trained on | fused delta |
|---|---|---|
| RTMDet | its own predicted boxes (3 seeds) | +0.092 +- 0.001 |
| FCOS   | its own predicted boxes (3 seeds) | +0.189 +- 0.002 |
| FCOS   | RTMDet's predicted boxes (transfer) | +0.040 |
| FCOS   | GT boxes (recipe-A)                 | +0.034 |

Per family, the RTMDet predicted-box recogniser gains +0.453 +- 0.011 (PSK), +0.537 +- 0.023
(QAM) and +0.373 +- 0.016 (ASK) over the same three seeds
(`work_dirs/repro/deployment/bridge_predhi_s{101,202,303}/summary.csv`).

**PSK gain of the predicted-box recogniser -- originally stated as +0.457, corrected to
+0.453 +- 0.011.** An intermediate revision of the manuscript had stated the predicted-box
recogniser's PSK gain as +0.457, which is the seed-101 value alone. It was replaced by the
three-seed mean +0.453 +- 0.011 from the summary files above; QAM (+0.537 +- 0.023) and ASK
(+0.373 +- 0.016) are stated on the same three-seed basis.

**The dose-response curve (§VI-D) -- originally reported as 0.643 -> 0.714 -> 0.869,
corrected to 0.632 (stage2-single) / 0.534 (combined) -> 0.699 -> 0.875.** The originally
reported 0.643 -> 0.714 -> 0.869 took a different metric from each of three runs. Measured
cleanly, each metric is monotone on its own: stage2-single accuracy 0.632 / 0.691 / 0.867,
combined fine accuracy 0.534 / 0.699 / 0.875. The originally reported 0.643 was a
stage2-single number labelled as combined. The manuscript now reports each metric under its
own label -- 40-epoch 0.632 stage2-single and 0.534 combined, recipe-B 0.699, recipe-A
0.875 -- with the differential-phase recogniser at 0.913 (originally 0.916), and the
deployment deltas refreshed to +0.028 fused (originally +0.024), +0.153 / +0.132 / +0.081 for
psk / ask / qam (originally +0.143 / +0.118 / +0.084) and +0.022 for differential phase
(originally +0.019). The oracle bound, 0.608, was unchanged.

## Reproducibility defects fixed (in the released code)

Found while running from the fresh clone; each is committed to `main`:

- **FCOS/ATSS were untrainable under mmcv-lite** (commit `89339ce`). mmdet dispatches a
  CUDA tensor's focal loss to `mmcv.ops.sigmoid_focal_loss`, whose compiled kernel mmcv-lite
  does not ship, so every `FocalLoss` head died in the first backward pass. RTMDet never hits
  it. `patch_focal_loss_for_mmcv_lite()` routes those heads to mmdet's own
  `py_sigmoid_focal_loss`, the same computation. The paper's own FCOS/ATSS numbers were
  produced by a one-line edit of the same effect made directly in a site-packages install --
  an edit that was never in the repository.
- **One truncated raw-IQ scene aborted a whole crop-cache build** (commit `88c02ff`). A single
  half-written `.npz` (1 of 2964 in the test set) raised deep inside zipfile with no file
  name. `load_raw_iq` now falls back to the decoded `.npy` cache, or names the scene if it
  cannot.

## Cross-detector taxonomy sweep

The predicted-box recipe is not tied to RTMDet or FCOS. We ran the whole per-detector chain --- train the
detector, dump its predicted boxes on train and test, build a count-matched (174,136) predicted-box crop
cache, train three recognizers (seeds 101/202/303), bridge and deploy --- on thirteen detectors spanning the
anchor-free, anchor-based, adaptive, dense, two-stage, multi-stage, set-prediction, and DETR families. Every
one gains; the paper reports this as Table `tab:taxonomy` and the per-family values are in
`taxonomy-results.csv`.

Configs are the nine `*_stft3_memmap_resize512.py` added here (`cascade_rcnn`, `faster_rcnn`,
`conditional_detr`, `dab_detr`, `deformable_detr`, `dino`, `gfl`, `retinanet`, `sparse_rcnn`), plus RTMDet,
FCOS, and ATSS already present. Each inherits its mmdet base with `_base_ = 'mmdet::...'` and loads
`mmdet_plugins` (the documented mmdet exception).

Per detector `<fam>`, from a fresh clone:

```bash
# 1. detector (20 ep). DETRs collapse at the uniform 5e-4 -- see the learning-rate note below.
python configs/detection_is_easy/run_mmdet_train_eval.py \
  --root data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass \
  --config configs/detection_is_easy/<fam>_stft3_memmap_resize512.py \
  --work-dir work_dirs/<fam>_det --epochs 20 --batch-size 4 --optimizer AdamW --lr 5e-4 \
  --seed 7 --require-tensor-stats
# 2. dump its predicted boxes on test and on train (same script, --eval-only --dump-results;
#    add --test-split train for the train dump)
# 3. count-matched predicted-box crop cache
python configs/detection_is_easy/build_pred_matched.py --fam <fam> \
  --baseline-pred work_dirs/<fam>_traindump/source_data/test_predictions.bbox.json \
  --work-dir work_dirs/<fam>_buildpred
# 4. three recognizers on that cache, then bridge --split test (see "Recognizer and deployment cells")
```

**Learning-rate deviations (recorded, not tuned).** Under the uniform lr `5e-4` every DETR variant collapses
to a zero-mAP detector (sane loss, degenerate queries). Lower the rate per family: RetinaNet and the
deformable-attention DETRs (Deformable-DETR, DINO) to `1e-4`; the plain DETRs (Conditional-DETR, DAB-DETR) to
`5e-5`. The op fallbacks in `run_mmdet_smoke.py` and `run_mmdet_train_eval.py` (RoIAlign, multi-scale
deformable attention, and NMS routed to `torchvision` / pure PyTorch) are what let the two-stage and DETR
detectors run under mmcv-lite without compiled ops.

## Third-round audit (2026-08-21/22): what was added and why

A third independent audit re-derived every number above from the archived `summary.csv`
files and closed the items the second round had left open. Nothing was retrained; no
learning rate, schedule, model, or split was changed. Everything below is inference or
analysis on archived artifacts, and each step is a released script.

**Same-prediction gap on three detector seeds (was: one seed).** The two missing
prediction dumps were regenerated by inference only (`run_mmdet_train_eval.py --eval-only
--dump-results`) from the archived `epoch_20.pth` checkpoints of the own-schedule medium
cell (seeds 20262811/17/27, batch 4), using the exact harness commit of the original
deployment dump (`88c02ff`). Re-dumping the deployment checkpoint itself reproduced the
original `test_predictions.bbox.json` byte for byte (identical SHA-256), so checkpoint +
commit determine these numbers. `same_pred_bootstrap.py` evaluates each dump twice with
`pycocotools` (`useCats=1` / `useCats=0`) and attaches a 2,000-resample scene-paired
bootstrap (one fixed seed, the same scene weights for both evaluations and all seeds; the
weighted re-accumulation reproduces `COCOeval.stats[0]` exactly at unit weights):

| detector seed | checkpoint | AP_cls | AP_loc | gap [95% scene bootstrap] |
|---|---|---|---|---|
| 20262811 (own schedule, batch 4) | `repro/axisB_own/bo_m_own_s20262811_bs4/epoch_20.pth` | 0.486 | 0.688 | 0.203 [0.197, 0.204] |
| 17 (own schedule, batch 4) | `repro/axisB_own/bo_m_own_s17/epoch_20.pth` | 0.496 | 0.703 | 0.207 [0.202, 0.209] |
| 27 (own schedule, batch 4) | `repro/axisB_own/bo_m_own_s27/epoch_20.pth` | 0.479 | 0.694 | 0.215 [0.210, 0.217] |
| 20262811 (deployment, batch 8) | `repro/deploy/deploy_m_own_s20262811/epoch_20.pth` | 0.465 | 0.712 | 0.246 [0.241, 0.247] |

Three-seed mean gap 0.208 +- 0.006 (sd over seeds), paired
bootstrap interval of the mean [0.203, 0.209]; every resample of every
seed is positive. Command:

```
python configs/detection_is_easy/same_pred_bootstrap.py \
  --annotation <memmap-root>/coco_multiclass/annotations/instances_test.json \
  --prediction 17=<work-dir>/source_data/test_predictions.bbox.json [--prediction SEED=PATH ...] \
  --resamples 2000 --seed 20260821 --output same_pred_bootstrap.json
```

**Routing rule fixed on validation (was: post-hoc on test).** The deployment detector's
validation-split predictions were dumped (`--test-split val`; val `bbox_mAP` 0.472
reproduced) and the three matched predicted-box recognizers were run through
`bridge.py bridge --split val` with the test-split flags. Rule, written before looking at
the output: route a family to IQ iff its three-seed mean per-family gain is positive.

| family | vision AP (val) | IQ - vision, seeds 101/202/303 | route |
|---|---|---|---|
| am | 0.730 | -0.248 / -0.243 / -0.251 | vision |
| ask | 0.208 | +0.335 / +0.340 / +0.323 | IQ |
| chirp | 0.727 | -0.145 / -0.149 / -0.151 | vision |
| fm | 0.856 | -0.314 / -0.313 / -0.311 | vision |
| fsk | 0.643 | -0.054 / -0.052 / -0.052 | vision |
| msk | 0.788 | -0.038 / -0.042 / -0.043 | vision |
| ofdm | 0.348 | -0.348 / -0.348 / -0.348 | vision |
| psk | 0.262 | +0.439 / +0.419 / +0.436 | IQ |
| qam | 0.124 | +0.465 / +0.461 / +0.443 | IQ |

Fused validation AP 0.463 -> 0.540 / 0.541 / 0.541
(+0.077 / +0.078 / +0.078). The rule selects exactly PSK/ASK/QAM, so the
test-split routed numbers above are unchanged; they are now a single evaluation under a
validation-fixed rule, with the disclosure that the test split had been inspected before
the rule was written.

**Box-quality AUC, scene-grouped (was: in-sample point estimates).** `box_quality_auc_cv.py`
runs five-fold scene-grouped cross-validation with the L2 strength of the 11-feature
logistic model selected inside the training folds, plus a 2,000-resample scene bootstrap of
the out-of-fold AUC. On the oracle-correct subset: 40-epoch dump IoU
0.678 [0.660, 0.696] vs multivariate
0.686; recipe-A (120-epoch) dump IoU
0.564, energy contamination 0.583,
multivariate 0.605. No scalar is a reliable
predictor; the archived table's values come from the 40-epoch dump (`box_quality_oracle.jsonl`,
135,434 / 60,529 rows), not the recipe-A dump used for the box-error statistics.

**Numbers corrected in this round.** The RTMDet predicted-box gain is **+0.092 +- 0.001**
(mean of the per-seed differences 0.0930/0.0922/0.0919); the earlier **+0.093** was the
difference of the rounded absolute values 0.567 - 0.474. The SNR diagnostic denominator was
recovered from the generator metadata (first 2,000 test scenes: 7,040 signals, 6,668 in
[-5, 35) dB, 179 below, 193 at or above 35 dB; 9 scenes / 10 signals absent from the matched
dump). Two historical negative-result values (weighted box fusion +0.0004, 40-epoch
differential phase +0.003 +- 0.003) could not be traced to any log and are no longer reported.

**Clean-clone gate finding.** Installing `requirements/detection_is_easy.txt` into a fresh
Python 3.10 venv with current pip/setuptools built 0/9 release configs: mmengine 0.10.7
resolves `mmdet::` bases through `pkg_resources`, which setuptools 8x no longer ships; earlier
gates passed only because a system-level `pkg_resources` leaked in. The requirements file now
pins `setuptools==59.6.0` (the campaign environment's version) and the 9/9 build is reproduced
from a clean venv.

**Release checker.** `tools/misc/check_paper.py` was probed with 96 adversarial cases and
hardened: `pkg==x.*`, `===`, compound specifiers and editable installs are not pins;
in-repository `-r`/`-c` includes are followed; `runtime_check` must be `{python}` plus an
in-repository script; `Co-authored-by` anywhere in a message is rejected and commit records
are read one commit at a time; POSIX-style `//<host>/<share>` roots, `smb://`-style URLs,
suffix-less configuration files and paths hidden inside placeholders or `$math$` are scanned.
The convention docs (`docs/adding_a_new_paper*.md`) describe the same rules.

## Provenance

Per-cell CSVs, the pooled comparison, and every run's `run_info.json` (with the literal
command line in `argv`, the git commit, and the two run-time contract flags) are archived
alongside this record. The paper-value reference table is
`configs/detection_is_easy/paper_values.csv`; regenerate the comparison with:

```
python configs/detection_is_easy/collect_repro_results.py \
  --root work_dirs/repro --markdown reports/repro_cells.md \
  --reference configs/detection_is_easy/paper_values.csv
```
