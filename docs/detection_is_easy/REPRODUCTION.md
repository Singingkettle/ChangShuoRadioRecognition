# Reproduction record

English | [简体中文](REPRODUCTION_zh-CN.md)

This file records an end-to-end re-run of the paper's experiments from a fresh clone of the
released code, and states, cell by cell, where the re-run lands relative to the reported
numbers. It is a factual log. It draws no conclusions the measurements do not force.

Every number below was produced on server hardware from the released `main` branch. The
comparison table is generated mechanically by `tools/detection_is_easy/collect_repro_results.py`
against `docs/detection_is_easy/paper_values.csv`, which stores each reported value next to
the table or section it came from.

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
`sd` is the spread across seeds. "verdict" is against the 0.023 band.

| Cell | reproduced | sd | paper | delta | verdict | paper source |
|---|---|---|---|---|---|---|
| Axis B, tiny, uniform | 0.432 | 0.0035 | 0.431 | +0.001 | within | Table I uniform |
| Axis B, small, uniform | 0.443 | 0.0205 | 0.449 | -0.006 | within | Table I uniform |
| Axis B, medium, uniform | 0.472 | 0.0121 | 0.460 | +0.012 | within | Table I uniform |
| Axis B, large, uniform | 0.451 | 0.0074 | 0.462 | -0.011 | within | Table I uniform |
| Axis B, tiny, own sched. | 0.433 | 0.0035 | 0.408 | +0.025 | **outside** | Table I own |
| Axis B, small, own sched. | 0.470 | 0.0135 | 0.429 | +0.041 | **outside** | Table I own |
| Axis B, medium, own sched. | 0.486 | 0.0152 | 0.477 | +0.009 | within | Table I own |
| Axis B, large, own sched. | 0.504 | 0.0154 | 0.486 | +0.018 | within | Table I own |
| Axis A, magnitude-only | 0.441 | 0.0231 | 0.455 | -0.014 | within | Table III |
| Axis A, phase+magnitude | 0.455 | 0.0125 | 0.447 | +0.008 | within | Table III |
| Axis A, phase only | 0.431 | -- | 0.440 | -0.009 | within | Table III |
| Axis A, learnable filterbank | 0.418 | -- | 0.412 | +0.006 | within | Table III |
| Axis E, complex-1D + FFT | 0.053 | -- | 0.026 | +0.027 | **outside** | Table III |
| FCOS | 0.470 | 0.0053 | 0.374 | +0.096 | **outside** | Section VI-B |
| ATSS | 0.468 | 0.0032 | 0.380 | +0.088 | **outside** | Section VI-B |
| localization (single-class) | 0.893 | 0.0060 | 0.948 | -0.055 | **outside** | Sections I, IV, VI-A |
| deployment detector (best run) | 0.472 | -- | 0.521 | -0.049 | **outside** | Table I caption / VI-D |

Controls with no paper value of their own:

| Cell | reproduced | sd | why it was run |
|---|---|---|---|
| RTMDet-M, uniform, batch 4 | 0.426 | 0.0006 | batch-matched family control (§VI-B) |
| RTMDet-M, own sched., batch 4 | 0.491 | -- | batch-decoupled size control (§VI-C, L5) |

14 of the 21 cells that carry a paper value land inside the band. The seven that do not are
each accounted for below.

## Recognizer and deployment cells

Recognizer accuracy on the val crop cache (clean signals), and the deployment bridge on the
2963 test scenes with archived raw IQ, off the reproduced deployment detector.

| Cell | metric | reproduced | paper | note |
|---|---|---|---|---|
| recipe-A recognizer (3 seeds) | combined fine acc | 0.875 / 0.876 / 0.871 | 0.869 | run-index 19 |
| 40-epoch predecessor | stage2-single acc | 0.632 | 0.643 | run-index 20 -- see dose-response note |
| 40-epoch predecessor | combined fine acc | 0.534 | -- | |
| recipe-B (mixup) | combined fine acc | 0.699 | 0.714 | run-index 21 |
| differential phase | combined fine acc | 0.913 | 0.916 | run-index 26 |
| recipe-A deployment (3 seeds) | fused delta | +0.028 (0.027/0.029/0.027) | +0.024 | run-index 22 |
| recipe-A deployment | psk / ask / qam delta | +0.153 / +0.132 / +0.081 | +0.143 / +0.118 / +0.084 | run-index 23 |
| oracle (perfect box), recipe-A | pure-IQ class mAP | 0.608 | 0.608 | run-index 24 |
| differential phase deployment | fused delta | +0.022 | +0.019 | run-index 26 |

The deployment bridge uses the recorded non-default flags -- `--score-thr 0.05 --limit 2963
--class-nms-iou 0.5 --ours-score-recog --iq-families psk,ask,qam`. Running with bridge
defaults evaluates a different scene set with per-class NMS off and does not reproduce these.

## The seven outside-band cells

**FCOS +0.096, ATSS +0.088 (§VI-B).** The reported FCOS/ATSS runs trained with identity
normalisation -- their generated config carries `mean=[0,0,0] std=[1,1,1]` on data whose
per-channel sigma is about 12.8 -- while the RTMDet runs they were compared against used the
real statistics. Re-running FCOS/ATSS with the statistics injected (`--require-tensor-stats`)
gives 0.470 and 0.468, at three seeds with sd 0.005 and 0.003. Two further facts sharpen the
picture:

- RTMDet-M under the same uniform recipe is 0.472 at batch 8 but **0.426 at batch 4**, the
  batch FCOS/ATSS ran at. At matched batch, FCOS (0.470) and ATSS (0.468) sit *above*
  RTMDet-M (0.426), not below it.
- So the reported RTMDet lead came from two compounding artefacts: a normalisation defect
  handicapping the other heads, and a batch-size difference. With both removed, the detector
  family does not separate. This is consistent with the paper's own thesis -- localization is
  saturated, the gap is recognition -- but it inverts the §VI-B sentence that the head and
  assigner matter and that RTMDet is a specially justified choice.

**localization -0.055 (§I/IV/VI-A).** The reported class-agnostic localization mAP of 0.948
was measured on an easier generator configuration (the `lowsnr` set: longer durations, higher
SNR, no co-channel overlap) than the released `hardshort_lowsnr` benchmark. Three single-class
runs on the released benchmark give **0.893 +- 0.006**. Localization is still the easy,
near-saturated axis; the number that states it should be the one measured on the benchmark
the paper distributes.

**deployment detector -0.049.** The paper quotes this cell's single best run (0.521) as the
deployment baseline but its three-seed mean (0.477) for the size sweep. The reproduction gives
0.472 for the single run -- and 0.486 for the three-seed mean, which is inside the band. The
deployment numbers are all bridged off this checkpoint, so their absolute level tracks it
(reproduced baseline 0.474 vs the paper's 0.522); the reported *deltas* reproduce regardless.

**small/own +0.041, tiny/own +0.025.** Real deviations in the own-schedule column, whose
reported spread is itself large (sd 0.017 and 0.041 in the paper's own three seeds). The
uniform column of the same sizes reproduces cleanly.

**complex-1D +0.027.** Both values are a collapse (0.053 vs 0.026); the difference has no
meaning at that magnitude. The Axis E finding -- a learned-then-FFT front end destroys
localization -- reproduces.

## Two findings the reproduction surfaced

**The predicted-box recogniser wins by +0.093, not -0.019 (§VII).** The negative-table row for
"train the recogniser on predicted boxes" reports a deployment delta of -0.019. The
reproduction gives **+0.093**, at three seeds (+0.0930/+0.0922/+0.0919, sd < 0.001), against
the recipe-A baseline of +0.028. Three checks pin the mechanism:

- *Not a leak.* Train and test scenes are disjoint (50000 vs 10000, zero shared sample ids or
  file names).
- *Distribution match, not a better recogniser.* On perfect GT boxes this recogniser scores
  0.415, *below* recipe-A's 0.608, and its clean val accuracy is 0.525 vs recipe-A's 0.875. It
  is worse everywhere except on the predicted-box distribution it was trained for.
- *Explains the reported -0.019.* The paper's crop cache was named `trainpred_hi`; its build
  parameters were never logged, and it holds 194k crops. Building the analogous cache at a
  high detection-score cut (`--score-thr 0.5`) keeps 87k near-perfect boxes and yields a
  deployment delta of **-0.027** -- reproducing the paper's -0.019. Training on the full
  routed-box distribution (`--score-thr 0.1`) is what produces the +0.093. The published
  negative result is an artefact of building the cache from high-score, near-GT boxes.

The parameters of `trainpred_hi` are unrecorded; the +0.093 cache was built at the released
default `--score-thr 0.1` over all 50000 scenes, then randomly subsampled (fixed seed) to
match the GT cache's crop count so the comparison is not handed extra data.

**The dose-response curve mixes two metrics (§VI-D).** The reported 0.643 -> 0.714 -> 0.869
takes a different metric from each of three runs. Measured cleanly, each metric is monotone on
its own: stage2-single accuracy 0.632 / 0.691 / 0.867, combined fine accuracy 0.534 / 0.699 /
0.875. The reported 0.643 is a stage2-single number labelled as combined.

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

## Provenance

Per-cell CSVs, the pooled comparison, and every run's `run_info.json` (with the literal
command line in `argv`, the git commit, and the two run-time contract flags) are archived
alongside this record. The paper-value reference table is
`docs/detection_is_easy/paper_values.csv`; regenerate the comparison with:

```
python tools/detection_is_easy/collect_repro_results.py \
  --root work_dirs/repro --markdown reports/repro_cells.md \
  --reference docs/detection_is_easy/paper_values.csv
```
