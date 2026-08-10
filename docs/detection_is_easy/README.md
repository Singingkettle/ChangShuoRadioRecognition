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

## Environment

```bash
pip install -r requirements/detection_is_easy.txt
```

That file pins the versions the paper ran on (torch 2.7.1+cu128, numpy 2.2.6, mmdet 3.3.0,
mmengine 0.10.7, torchsig 2.1.1) on Ubuntu with 8×RTX 4090.

**One choice changes your numbers: which mmcv you install.** Every reported result was
produced with `mmcv-lite` — mmcv *without* the compiled `_ext` CUDA ops. The harness detects
the missing extension and installs a pure-PyTorch NMS fallback (`maybe_stub_mmcv_ext()` in
`run_mmdet_smoke.py`), and every run records this as `used_mmcv_lite_stub: true` in its
`run_info.json`; all 268 runs carrying that field have it set. Installing full CUDA mmcv is
supported and faster, but swaps in a different NMS implementation, so expect small
differences. Pick one and keep it fixed for the whole comparison.

`torchsig` is only needed to regenerate the dataset. The pin matters there: the generator's
class taxonomy is what defines the 57 classes.

## Data

Synthetic wideband captures from TorchSig, generated with a custom configuration:
**50 000 / 5 000 / 10 000 train/val/test scenes** (65 000 total — the "65k" in the directory
name is the total, not the training split), 262 144 complex samples per scene at 10 MHz,
1–6 signals per scene, 57 classes.

Two properties of this configuration are what make the task hard, and both are in the name of
the dataset directory. `hardshort`: signal durations are 0.5 %–25 % of the observation, so
each emission occupies a small patch of the spectrogram. `lowsnr`: per-signal SNR is drawn
from −20 dB to +10 dB, stratified into three equal buckets.

The generated assets are large (~191 GB packed STFT memmap, ~128 GB raw IQ) and are not
shipped. Six commands rebuild them, with the paper's exact parameters:

```bash
cd <repo-root>
DATA=data                       # or an NVMe scratch path
RAW=$DATA/torchsig_hardshort_lowsnr_iq_65k_nvme
MM=$DATA/torchsig_hardshort_lowsnr_stft3_memmap

# 1) raw IQ scenes + per-signal metadata  (the slow step)
python tools/detection_is_easy/prepare_torchsig_iq_stratified.py \
  --out-root $RAW \
  --train 50000 --val 5000 --test 10000 \
  --num-iq-samples 262144 --sample-rate 10000000 \
  --num-signals-min 1 --num-signals-max 6 --impairment-level 0 \
  --fft-size 512 --stft-fft 512 --stft-hop 512 \
  --duration-min-frac 0.005 --duration-max-frac 0.25 \
  --bandwidth-min-frac 0.0125 --bandwidth-max-frac 0.49 \
  --center-freq-min-frac -0.45 --center-freq-max-frac 0.45 \
  --snr-buckets '-20,-10;-10,0;0,10' \
  --cochannel-overlap-probability 0.35 \
  --fast-snr-update \
  --seed 20260640

# 2) complex STFT tensors [2,F,T] + COCO annotations
python tools/detection_is_easy/export_complex_stft_coco_from_raw.py \
  --src-root $RAW --out-root $MM --stft-fft 512 --stft-hop 512

# 3) 3-channel [real, imag, log-magnitude] feature tensors  (SEPARATE --out-root)
python tools/detection_is_easy/make_stft_feature_tensor_from_complex.py \
  --src-root $MM/coco --out-root ${MM}_stft3 --mode realimag_logpower3ch --workers 8

# 4) pack into the memmap the fast training path reads
python tools/detection_is_easy/pack_coco_tensors_to_memmap.py \
  --kind tensor --src-coco ${MM}_stft3/coco --out-root $MM --splits train,val,test --workers 8

# 5) single-class ("signal") annotations -- the class-agnostic localization task
python tools/detection_is_easy/export_raw_coco_from_metadata.py \
  --src-root $RAW --out-root $MM --single-class

# 6) 57-class annotations -- the class-aware task
python tools/detection_is_easy/build_multiclass_coco.py \
  --dataset-dir $MM --out-dir $MM/coco_multiclass/annotations --splits train,val,test
```

After this, `$MM/coco/` holds the single-class annotations (used with `--root $MM/coco`) and
`$MM/coco_multiclass/` the 57-class ones (everything else).

### Five things about this chain that will silently cost you the benchmark

**The SNR range and the buckets are one choice, not two.** Pass `--snr-buckets` *or*
`--snr-db-min/--snr-db-max`, never a conflicting pair — the tool now aborts on a mismatch.
Earlier revisions derived the range from the buckets unconditionally, so a command that passed
`--snr-db-min -20 --snr-db-max 10` without buckets silently generated over the *default*
−10…+40 dB span in five buckets. That is a different, much easier benchmark, produced with no
error and no warning. If you give only the range, buckets are cut into `--snr-num-buckets`
(default 3) equal parts; the resolved plan is printed at startup — read it.

**`--fast-snr-update` changes both the physics and the random stream.** It replaces TorchSig's
per-signal spectrogram refinement with commanded time-domain power scaling, and it draws
`snr_db` from the dataset generator's RNG. The paper used it (`summary.json` records
`fast_snr_update: true`). Omitting it gives a different dataset from the same seed.

**Step 3 must not write into its own input.** With `--src-root $MM/coco --out-root $MM` the
output resolves back to `$MM/coco`, and the step would overwrite the `[2,F,T]` tensors it is
reading with `[3,F,T]` ones — unresumable, and on a re-run it fails on every already-converted
file. The tool refuses this now; give a separate `--out-root` as above.

**Step 6 links `coco/<split>/` into the multiclass root, and that link is load-bearing.** The
harness picks the `tensors/` data prefix by testing whether `<root>/<split>/tensors` exists, and
the memmap loader reads the split name back out of that path. A multiclass root holding only
`annotations/` resolves the split to `images` and fails with
`FileNotFoundError: .../memmap/images.npy`.

**Sharding was not released.** The paper's corpus was generated as ten shards merged by
hardlink (`summary.json` records `merge_mode: hardlink` and `source_shards: shard_000..009`);
the driver that produced and merged them is not part of this repository, and per-shard seeds
were never recorded. The commands above generate monolithically. The result is a corpus drawn
from the same distribution with the same generator settings, **not the same realization** — a
ten-shard run and a single run consume the RNG differently. Acceptance is therefore statistical,
via the equivalence check below, not a checksum.

### Check what you built before you train on it

```bash
python tools/detection_is_easy/validate_coco.py --root $MM
```

Confirm at minimum: 57 categories with ids 0–56 identical across splits; 50 000/5 000/10 000
images; per-signal `snr_db` spanning −20…+10 dB and not −10…+40; box widths and heights
consistent with `duration_frac ∈ [0.005, 0.25]` and `bandwidth_frac ∈ [0.0125, 0.49]`;
`summary.json` carrying `stft_tensor_stats` with three channels; and `memmap/<split>.npy`
row count equal to the annotation image count. `summary.json` also records a `provenance`
block (torchsig version, git commit, seed, argv) — quote it when reporting a reproduction.

If the data does not sit under `<repo-root>/data/`, pass `--memmap-root` / `--raw-root` to the
training harness and set `IQDET_MEMMAP_ROOT` / `IQDET_RAW_ROOT` / `IQDET_CACHE_ROOT` for
`bridge.py`. `--root` alone only moves the annotations.

## Reproducing a number

Three stages. Each produces the input the next one needs.

### Stage 1 — detector

**Training and dumping predictions are two separate calls.** `--dump-results` only sets the
test evaluator's output prefix; the training call never runs the test loop, so it writes no
predictions. Train first, then re-invoke in `--eval-only` mode against the checkpoint:

```bash
# train (this is the deployment detector: the run every bridged number is computed from)
python tools/detection_is_easy/run_mmdet_train_eval.py \
  --root $MM/coco_multiclass \
  --config configs/detection_is_easy/rtmdet_m_stft3_tensor_memmap_resize512.py \
  --work-dir work_dirs/baseline_mc_rtmdet_m_20e_seed20262811 \
  --epochs 20 --batch-size 8 --optimizer config --seed 20262811

# then dump test predictions from the trained checkpoint
python tools/detection_is_easy/run_mmdet_train_eval.py \
  --root $MM/coco_multiclass \
  --config configs/detection_is_easy/rtmdet_m_stft3_tensor_memmap_resize512.py \
  --work-dir work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump \
  --eval-only --checkpoint work_dirs/baseline_mc_rtmdet_m_20e_seed20262811/epoch_20.pth \
  --dump-results
# -> work_dirs/..._testdump/source_data/test_predictions.bbox.json
```

`--work-dir` is required and has no default. `--optimizer config` keeps the config's own AdamW
(lr 1e-4); passing `--optimizer AdamW --lr 5e-4` instead selects the *other* recipe — see the
table below, where the two are the "uniform recipe" and "own schedule" columns.

### Stage 2 — recognizer

Cache the channelized crops, then train. The paper's recognizer was trained by `bridge.py`,
which is what produces a checkpoint `bridge.py bridge` can load:

```bash
for s in train val test; do
  python tools/detection_is_easy/bridge.py build --split $s --L 1024
done   # -> work_dirs/returniq_cache/{train,val,test}_L1024.npz

python tools/detection_is_easy/bridge.py train-hier \
  --train-cache work_dirs/returniq_cache/train_L1024.npz \
  --val-cache   work_dirs/returniq_cache/val_L1024.npz \
  --out work_dirs/returniq_cache/recognizer_hierrcpA_s101.pth \
  --epochs 120 --label-smooth 0.1 --cosine --ema 0.999 --aug-cfo 0.02 --seed 101
```

**`train-hier`'s defaults are the paper's negative result, not its headline.** They are 40
epochs, no label smoothing, no cosine schedule, no EMA — exactly the under-trained recognizer
that produced the deployment tie the paper nearly published as a structural limit (clean
accuracy 0.643 vs 0.869 for the recipe above). The five flags in that command are the finding.

The same recognizer is also available as a first-class CSRR model, trained the usual way:

```bash
python tools/train.py configs/detection_is_easy/returniq_resnet1d_iq_120e_wideband.py
```

Use this path to study the architecture inside CSRR. Use `bridge.py train-hier` to reproduce
the paper: the two save different checkpoint formats, and `bridge.py bridge` reads the latter.

### Stage 3 — deployment bridge

```bash
python tools/detection_is_easy/bridge.py bridge \
  --split test \
  --baseline-pred work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump/source_data/test_predictions.bbox.json \
  --hier-model work_dirs/returniq_cache/recognizer_hierrcpA_s101.pth \
  --L 1024 --score-thr 0.05 --limit 2963 --class-nms-iou 0.5 \
  --ours-score-recog --iq-families psk,ask,qam
```

**Do not run this with defaults.** Four of those flags differ from the defaults, and each one
alone changes the answer:

| flag | default | paper | what the default does |
|---|---|---|---|
| `--score-thr` | `0.0` | `0.05` | keeps near-zero-score detections, flooding both methods |
| `--limit` | `0` (all) | `2963` | scores a different scene set, so numbers are not comparable to the paper |
| `--class-nms-iou` | `1.0` | `0.5` | **disables** per-class NMS; duplicate boxes inflate both sides asymmetrically |
| `--ours-score-recog` | off | on | ranks routed detections by detection score alone, discarding recognition confidence |

`oracle` gives the perfect-box upper bound; `diag-quality` writes the per-detection dump that
Fig. 2 is built from.

## Which cell is which: config, flags, seed, expected value

Every headline number, with what produces it. mAP values are `coco/bbox_mAP` on the **val**
split unless noted. All detector rows use `--root $MM/coco_multiclass` and
`run_mmdet_train_eval.py`; "uniform" = `--optimizer AdamW --lr 5e-4`, "own" = `--optimizer config`.

The class-agnostic (localization-only) task needs no separate config: point `--root` at the
single-class annotations (`$MM/coco`) and the harness reads one category from them and sets
`num_classes` accordingly.

| paper cell | config | recipe / flags | seed | expected | tol. |
|---|---|---|---|---|---|
| Localization is easy (class-agnostic) | `rtmdet_m_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8`, **`--root $MM/coco`** | any | ≈0.915 | ±0.01 |
| Tab. I tiny / uniform | `rtmdet_tiny_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | 7 | 0.420 | ±0.02 |
| Tab. I small / uniform | `rtmdet_s_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | 7 | 0.438 | ±0.02 |
| Tab. I medium / uniform | `rtmdet_m_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | 7 | 0.447 | ±0.02 |
| Tab. I large / uniform | `rtmdet_l_stft3_tensor_memmap_resize512.py` | uniform, `--batch-size 8` | 7 | 0.451 | ±0.02 |
| Tab. I tiny / own | `rtmdet_tiny_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | 20262811 | 0.379 | ±0.04 |
| Tab. I small / own | `rtmdet_s_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | 20262811 | 0.449 | ±0.03 |
| Tab. I medium / own **(deployment detector)** | `rtmdet_m_stft3_tensor_memmap_resize512.py` | own, **`--batch-size 8`** | 20262811 | 0.521 | ±0.04 |
| Tab. I large / own | `rtmdet_l_stft3_tensor_memmap_resize512.py` | own, `--batch-size 4` | 20262811 | 0.482 | ±0.02 |
| Tab. III magnitude-only (phase out) | `rtmdet_m_rawiq_fourier_logmag2ch_resize512.py` | uniform, `--batch-size 6` | 7 | 0.476 | ±0.03 |
| Tab. III phase + magnitude | `rtmdet_m_raw_iq_filterbank_hardshort_resize512.py` | uniform, `--batch-size 6` | 7 | 0.446 | ±0.02 |
| Tab. III phase only | `rtmdet_m_rawiq_fourier_realimag_resize512.py` | uniform, `--batch-size 6` | 7 | 0.440 | ±0.02 |
| Tab. III learnable filterbank | `rtmdet_m_rawiq_learnable_realimag_logmag_resize512.py` | uniform, `--batch-size 6` | 7 | 0.412 | ±0.02 |
| Tab. III complex-1D + FFT bridge | `rtmdet_m_complexiq1d_fftbridge_resize512.py` | uniform, `--batch-size 6` | 7 | 0.026 | collapse |
| §VI-B FCOS anchor | `fcos_stft3_memmap_resize512.py` | uniform, `--batch-size 4` | 7 | 0.374 | ±0.02 |
| §VI-B ATSS anchor | `atss_stft3_memmap_resize512.py` | uniform, `--batch-size 4` | 7 | 0.380 | ±0.02 |
| §IV-A recognizer, recipe A | — | `train-hier --epochs 120 --label-smooth 0.1 --cosine --ema 0.999 --aug-cfo 0.02` | 101/202/303 | 0.869 clean accuracy | ±0.006 |
| §VI-D recognizer, 40-epoch predecessor | — | `train-hier` **defaults** (`--epochs 40 --aug-cfo 0.02`) | 101 | 0.643 | ±0.01 |
| §VI-D deployment, vision → routed | — | the Stage-3 command above | 101/202/303 | 0.522 → 0.546 (+0.024) | ±0.002 on the delta |
| §VI-D per-family PSK / ASK / QAM | — | same command | 101/202/303 | +0.143 / +0.118 / +0.084 | ±0.011 / ±0.008 / ±0.012 |
| §VI-D oracle (perfect box) | — | `oracle --with-oracle --limit 2000 --score-thr 0.05` | 101 | 0.420 → 0.608 | ±0.01 |

Two cautions about that table.

The size sweep: the medium/own cell is the best single run in the sweep (0.521), while its
3-seed mean is 0.477 ± 0.039. The paper reports both, and the spread is why the complexity
conclusion rests on the uniform column.

The localization row: the paper quotes 0.948 for class-agnostic localization. That value was
measured on an earlier, easier generator configuration (signal durations 5–100 % of the
observation, SNR −10 to +20 dB, no co-channel overlap) rather than on the hardshort-lowsnr
benchmark released here (durations 0.5–25 %, SNR −20 to +10 dB, 35 % co-channel overlap). On
this benchmark the matched STFT3 single-class run reaches about 0.915, and the best result of
any recipe is 0.945. The conclusion is unchanged — localization is saturated either way, and
Fig. 2's localization recall of ≈0.99 is measured on this benchmark's test split — but expect
≈0.915, not 0.948, when you run the command above.

## Two different metrics are both called "class-aware mAP"

They are not interchangeable, and mixing them is the fastest way to conclude the paper is
wrong.

- **`coco/bbox_mAP`** — mmdet's `CocoMetric`, averaged over the 57 categories, on the **val**
  split. This is the detector ablation metric: 0.447, 0.521, 0.476, and every other cell in the
  table above.
- **`class_aware_detection_map`** — the time-frequency IoU metric in `iqdet_metrics.py`,
  computed over the first 2963 **test** scenes. This is the deployment metric: 0.522 → 0.546.

The deployment baseline (0.522) and the detector's val mAP (0.521) are close by coincidence.
They are different metrics on different splits.

## What counts as a successful reproduction

Training is **not** deterministic: the harness sets `randomness = dict(seed=..., deterministic=False)`,
so cuDNN picks non-deterministic kernels. Even with the same seed and the same machine, a
detector re-run lands within roughly ±0.02 class-aware mAP; across seeds the spread is larger
(the size-sweep cells above carry their measured 2–3-seed standard deviations).

Reproduce the *conclusions*, which are robust, rather than chasing the third decimal:

1. Class-agnostic localization ≈ 0.95 while 57-class mAP ≈ 0.45. The gap is the paper.
2. Magnitude-only and phase+magnitude tie. Neither wins by more than the seed spread.
3. The learnable front end loses to the frozen one.
4. mAP is flat across tiny → large under a fixed recipe.
5. Routing PSK/ASK/QAM boxes back to IQ gains ≈ +0.02 overall and ≈ +0.1 on PSK.
6. The recognizer's 120-epoch recipe beats its 40-epoch predecessor by ≈ +0.23 clean accuracy.
   This is a budget effect, not an architecture effect.

If (1)–(6) hold, the reproduction succeeded even if individual cells differ in the second
decimal.

## Figures

```bash
python tools/detection_is_easy/make_figs.py            # Figs. 1, 2, 4, 5 -> figs/*.pdf
python tools/detection_is_easy/render_example.py \
  --ann $MM/coco_multiclass/annotations/instances_test.json \
  --raw $RAW/raw/test \
  --pred work_dirs/baseline_mc_rtmdet_m_20e_seed20262811_testdump/source_data/test_predictions.bbox.json
```

`make_figs.py` is self-contained: it reads only the committed `snr_data.csv` beside it, which
is the output of `analyze_snr_stratified.py` on the recipe-A diagnostic dump. Regenerate that
CSV with:

```bash
python tools/detection_is_easy/bridge.py diag-quality \
  --hier-model work_dirs/returniq_cache/recognizer_hierrcpA_s101.pth \
  --baseline-pred <the test dump> --L 1024 --score-thr 0.05 --with-oracle --limit 2000 \
  --out work_dirs/returniq_cache/box_quality_oracle_rcpA.jsonl
python tools/detection_is_easy/analyze_snr_stratified.py \
  --jsonl work_dirs/returniq_cache/box_quality_oracle_rcpA.jsonl --limit 2000
```

## Documented deviations / notes

- **Block-SNR correction.** All SNR-stratified results use
  `block_snr = snr_db + 10*log10(1/(tf*ff))`, where `tf` and `ff` are the signal's time and
  frequency occupancy. The generator's `snr_db` is a whole-observation average and understates
  visibility by a median of about +16.7 dB. Do not label results "low-SNR" on the raw axis.
- **mmcv `_ext` stub.** See the environment section: the paper's numbers come from the
  pure-PyTorch NMS fallback, recorded per run in `run_info.json`.
- **Normalisation statistics.** The raw-IQ filterbank configs carry per-channel mean/std taken
  from the offline STFT3 statistics rather than recomputed on their own front-end output. They
  began as a placeholder and were never revised, so they are what every reported number for
  those cells was trained with. Treat them as part of the recipe and do not recompute them:
  both arms of the phase test share the same constants, so the comparison stays fair, but
  changing them changes the values.
- **The localization number.** See the caution under the reproduction table: the paper's 0.948
  comes from an earlier, easier generator configuration; this benchmark gives ≈0.915 for the
  matched run.
- **Synthetic provenance.** Classes, boxes, and SNR are generator ground truth; there is no
  measurement noise floor to hide behind — the recognition gap is structural, and the released
  configs make every number regenerable.

Licensed under the Apache License, Version 2.0.
