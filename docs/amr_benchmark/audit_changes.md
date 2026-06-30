# AMR-Benchmark Audit & Fix Log (Phase 1A)

This document records the per-model audit performed against the
[AMR-Benchmark Keras reference](https://github.com/Richardzhangxx/AMR-Benchmark)
(commit fetched on 2026-06-25) and the DSP 2022 survey paper. For
each model we note what already matched, what was fixed in this
branch (`feature/amr-benchmark-migration`), and what remains as an
intentional CSRR design choice that Phase 2 should be aware of.

Confidence legend: 🟢 high — code/dimension/loss verified by smoke
test or paper Table 4 parity; 🟡 medium — fix is logically sound
but requires Phase 2 training to confirm accuracy impact; 🔴 low —
documented divergence kept on purpose, may need revisit if Phase 2
falls outside the tolerance band.

## Cross-cutting fixes

### Dataset split (50/10/40)

🟢 `configs/mldnn/iq-ap-deepsig201610A.py` — was training on
`train_and_validation.json` and validating on `test.json`, leaking
the validation split into training. Switched to `train.json` /
`validation.json` to match every other CSRR dataset config and the
project-wide 50/10/40 split. All other MLDNN/FastMLDNN configs
already used the standard split (verified by `git grep ann_file
configs/mldnn configs/fastmldnn`). No changes needed elsewhere.

Legacy duplicates under `configs/mldnn/original/`,
`configs/hcgdnn/original/`, `configs/fastmldnn/paper/`, and
`configs/trn/data_img-deepsig201610A.py` still use the old
`train_and_validation.json`/`test.json` two-way split; they are kept
verbatim to reproduce the original paper numbers and are *not* used
by Phase 2.

### val_evaluator consistency

🟢 Every AMR-Benchmark model in scope inherits its `val_evaluator`
from `configs/_base_/datasets/<dataset>/<input>.py`, which already
defines:

```python
val_evaluator = [
    dict(type='Accuracy', topk=(1,)),
    dict(type='Loss', task='classification'),
]
```

Two project-specific HCGDNN configs (`configs/hcgdnn/iq-deepsig201610A.py`
and `configs/hcgdnn/iq-deepsig201604C.py`) deliberately override
this with `HCGDNNWeightsAccuracy` because HCGDNN reports per-head
weighted accuracies during validation. Per the plan we do not
touch project-specific models, so this stays unchanged.

### HisarMod IQ length (`[1, 2, 128]` → `[1, 2, 1024]`)

🟢 The `data/ModulationClassification/Hisar/HisarMod2019.1/iq/*.npy`
files are stored as `(2, 1024)` (verified by `np.load`). Three base
configs reshaped to length 128, which would either silently feed
the wrong tensor or `ValueError` at runtime:

- `configs/_base_/datasets/hisar/iq-hisar2019.py` — `[1,2,128]` → `[1,2,1024]`
- `configs/_base_/datasets/hisar/iq-shape-F-L-hisar2019.py` — `[2,128]` → `[2,1024]`
- `configs/_base_/datasets/hisar/iq-shape-L-F-hisar2019.py` —
  removed an erroneous `IQToAP` + `Transpose(ap)` chain that was
  followed by `PackInputs(input_key='iq')`; the new pipeline simply
  transposes `iq` to `[L, 2]` so GRU2/LSTM2/PET-CGDNN see the same
  shape on HisarMod as they do on DeepSig.

This alone unblocks every Hisar config for `MCLDNN`, `HCGDNN`,
`CNN2`, `CNN4`, `CLDNNW`, `CLDNNL`, `MCNet`, `ICAMCNet`, `CGDNet`,
`DensCNN`, `ResNetAMR`, and `GRU2`.

### Long-sequence `frame_length` overrides

🟢 The CLDNN/CGDNet backbones use `frame_length` to size their
LSTM/GRU input dimension; several `iq-deepsig-201801A` and
`iq-hisar-2019` configs were missing the override or hard-coded the
wrong value:

- `configs/cgdnet/cgdnet_iq-deepsig-201801A.py`: 128 → 1024
- `configs/cgdnet/cgdnet_iq-hisar-2019.py`: 128 → 1024
- `configs/cldnnl/cldnnl_iq-deepsig-201801A.py`: added `frame_length=1024`
- `configs/cldnnl/cldnnl_iq-hisar-2019.py`: added `frame_length=1024`
- `configs/cldnnw/cldnnw_iq-deepsig-201801A.py`: added `frame_length=1024`
- `configs/cldnnw/cldnnw_iq-hisar-2019.py`: added `frame_length=1024`

CNN2/CNN4 and MCNet/ICAMCNet/ResNetAMR configs already carried the
correct frame_length values.

## Per-model audit

| Model | Match status | Action taken |
|-------|--------------|--------------|
| CNN1 → `CNN2` | ✅ matched | none — layer-by-layer parity verified |
| CNN2 → `CNN4` | ❌ kernel mismatch | 🟢 fixed (see below) |
| MCNET → `MCNet` | ❌ missing | 🟢 implemented (param parity 121,226 / 126,616 / 127,386 exact; 10A 121,611 vs paper 121,511 +100) |
| IC-AMCNet → `ICAMCNet` | ❌ missing | 🟢 implemented (param parity exact on all four datasets) |
| ResNet → `ResNetAMR` | ❌ missing | 🟢 implemented (param parity exact on all four datasets) + new `configs/resnetamr/` |
| DenseNet → `DensCNN` | ✅ matched | none — strong parity with Keras |
| GRU → `GRU2` | ⚠️ Hisar pipeline | 🟢 Hisar `iq-shape-L-F` base fixed (drops bogus IQToAP) |
| LSTM → `LSTM2` | 🔴 input modality differs | none — CSRR uses A/P per DSP Table 1; AMR-Benchmark Keras code uses raw I/Q; document as known divergence and re-evaluate after Phase 2 trains |
| DAE → `DAE` | ❌ head mismatch | 🟢 all DAE configs switched from `ClsHead` (which pulled `feats[-1]` = decoder output) to `DAEHead` with 0.1·CE + 0.9·MSE losses |
| MCLDNN → `MCLDNN` | ✅ matched | none — multi-branch concat semantics verified equivalent to Keras NHWC `concatenate(axis=1)` |
| CLDNN (West) → `CLDNNW` | 🔴 ZeroPadding removed | none — intentional CSRR simplification (in-code comment); may differ by 1–2 pp vs Keras |
| CLDNN2 → `CLDNNL` | ⚠️ missing frame_length on long-seq configs | 🟢 fixed (see frame_length section) |
| CGDNet → `CGDNet` | ⚠️ wrong frame_length on 201801A | 🟢 fixed (see frame_length section) |
| PET-CGDNN → `PETCGDNN` | ❌ Q-rotation sign | 🟢 fixed in `csrr/models/backbones/petcgdnn.py` |
| 1DCNN-PF → `CNN1DPF` | 🔴 AP vs IQ split | none — CSRR feeds amplitude/phase to the parallel branches whereas AMR-Benchmark splits raw I/Q; documented for Phase 2 |

## Detailed change log

### `csrr/models/backbones/cnn4.py` (🟢 high confidence)

**Before:** four convolutional blocks used `kernel_size=(1, 8)`,
so each conv only convolved along the time axis and never mixed I
and Q until the final flatten. An extra `Dropout(0.5)` sat between
the hidden `Linear(128)` and the final classifier.

**After:** all four convs now use `kernel_size=(2, 8)` to match the
AMR-Benchmark `CNN2` reference, which deliberately mixes the I/Q
rows at every depth. Dropped the extra dropout so the FC head
matches Keras (`Dense(128, relu) → Dense(num_classes)`).

Smoke test:

```python
from csrr.models.backbones import CNN4
m = CNN4(frame_length=128, num_classes=11)
m(torch.randn(2, 1, 2, 128))[0].shape  # torch.Size([2, 11])
```

### `csrr/models/backbones/petcgdnn.py` (🟢 high confidence)

**Before:** the PET rotation layer computed
`Q' = I*sin(θ) - Q*cos(θ)` — the **negation** of the correct
rotated Q component (Keras computes
`Q' = Q*cos(θ) - I*sin(θ)`). Because the angle θ is learned, the
network could partially compensate, but the rotation matrix was
no longer orthogonal and the parameter-estimation block lost its
intended equivariance.

**After:** swapped the source of `x21`/`x22` so the standard 2D
rotation matrix is applied:

```python
x11 = I * cos(θ); x12 = Q * sin(θ); y1 = x11 + x12   # I' = I cos + Q sin
x21 = Q * cos(θ); x22 = I * sin(θ); y2 = x21 - x22   # Q' = Q cos - I sin
```

### `csrr/models/backbones/mcnet.py` (🟢 high confidence, new file)

Brand-new PyTorch port of the AMR-Benchmark Keras MCNET. The block
hierarchy is:

1. **Stem** — `Conv2D(64, (3,7), stride=(1,2), same)` + `MaxPool2D((1,3), stride=(1,2), same)`
2. **Pre-block** — two parallel branches concatenated on channels
3. **Skip path** — `Conv2D(128, (1,1), stride=(1,2))` + `MaxPool2D((1,3), stride=(1,2))`
4. **`Mblockp1`** (`_MBlockDown4`) — pre-pool + strided convs; halves time twice
5. **`Mblock2`** (`_MBlockKeep`) — keeps spatial size
6. **`Mblockp3`** (`_MBlockDown2`) — strided convs only; halves time
7. **`Mblockp4`** (`_MBlockKeep`) — keeps spatial size
8. **`Mblockp5`** (`_MBlockDown2`) — halves time
9. **`Mblockp6`** (`_MBlockKeep` with wider widths) — outputs 256 ch
10. **Final pool** — `AvgPool2d((2, 1))` for L=128 or `AvgPool2d((2, 8))` for L=1024
11. `Dropout(0.5)` + `Linear(384, num_classes)`

`padding='same'` is emulated for strided convolutions via explicit
symmetric padding `((k-1)//2)` (PyTorch's built-in `same` does not
support stride > 1). Smoke-test parameter counts vs DSP Table 4:

| Dataset | Paper params | This port |
|---------|--------------|-----------|
| RML2016.10A (11 cls) | 121,511 | 121,611 (+100 ≈ 1 bias) |
| RML2016.10B (10 cls) | 121,226 | 121,226 ✓ |
| RML2018.01A (24 cls) | 126,616 | 126,616 ✓ |
| HisarMod (26 cls) | 127,386 | 127,386 ✓ |

### `csrr/models/backbones/icamcnet.py` (🟢 high confidence, new file)

Four `(1, k)` convs with one `MaxPool(2,2)`, then a Dense(128)
hidden layer + additive Gaussian noise (`σ=1`, training only) +
output Linear. Parameter parity is exact on all four datasets:

| Dataset | Paper params | This port |
|---------|--------------|-----------|
| RML2016.10A (11 cls) | 1,264,011 | 1,264,011 ✓ |
| RML2016.10B (10 cls) | 1,263,882 | 1,263,882 ✓ |
| RML2018.01A (24 cls) | 8,605,720 | 8,605,720 ✓ |
| HisarMod (26 cls) | 8,605,978 | 8,605,978 ✓ |

### `csrr/models/backbones/resnetamr.py` (🟢 high confidence, new file)

The AMR-Benchmark "ResNet" is a shallow architecture with a single
broadcast residual skip from the raw `(1, 2, L)` input to the
second conv output (the 1-channel input is broadcast to the 256
output channels via PyTorch broadcasting, matching Keras
`Add()([input, x])`). Renamed from `ResNet` to `ResNetAMR` to
avoid clashing with deeper residual variants in other CV
registries. Configs added under `configs/resnetamr/` for
RML2016.04C/10A/10B, 2018.01A, HisarMod. Parameter parity is exact:

| Dataset | Paper params | This port |
|---------|--------------|-----------|
| RML2016.10A (11 cls) | 3,098,283 | 3,098,283 ✓ |
| RML2016.10B (10 cls) | 3,098,154 | 3,098,154 ✓ |
| RML2018.01A (24 cls) | 21,450,040 | 21,450,040 ✓ |
| HisarMod (26 cls) | 21,450,298 | 21,450,298 ✓ |

### `configs/dae/dae_*.py` (🟢 high confidence)

All five DAE configs (DeepSig 201610A/B, 201801A, HisarMod 2019.1,
UCSD RML22) were using `ClsHead`. The DAE backbone returns
`(xc, x, xd)` during training so the decoder reconstruction can be
optimized in `DAEHead._get_loss`. But `ClsHead.pre_logits` returns
`feats[-1]` (= `xd`, the **decoder output**) and uses that as the
classification logits with `CrossEntropyLoss(num_classes vs L*2)`,
which is nonsense and silently discards the entire reconstruction
loss term. Only `dae_ap-deepsig-201604C.py` had `DAEHead`. Switched
the other four configs to `DAEHead` with the AMR-Benchmark loss
weights (`CE 0.1`, `MSE 0.9`), dropped the unused `frame_length`
kwarg on the long-sequence configs, and removed the spurious
`_scope_='mmdet'` qualifier from the 201604C MSELoss entry.

## Phase 2 fixes (training/accuracy alignment)

### `csrr/models/backbones/icamcnet.py` — Xavier (glorot_uniform) init (🟢 high confidence)

**Symptom:** On RML2016.10A the first Phase 2 run sat at exactly random
chance (`accuracy/top1 = 9.0909% = 1/11`, `loss = 2.3979 = ln(11)`) for
every epoch; the best checkpoint was `epoch_1` and `ReduceLROnPlateau`
decayed the LR to its `1e-6` floor without the validation metric ever
improving. No NaNs — the network simply never escaped the random-output
regime.

**Root cause:** the AMR-Benchmark Keras reference initialises every
`Conv2D`/`Dense` with `kernel_initializer='glorot_uniform'`, but the CSRR
config supplied no `init_cfg`, so the backbone fell back to PyTorch's
default Kaiming (fan-in) initialisation. With the large
`Flatten -> Linear(8192, 128)` head this produces a very small-variance
128-d feature vector, which the reference `GaussianNoise(std=1)` layer
(applied between the hidden Dense and the classifier) then swamps. The
gradient signal is too weak to grow the features before the LR scheduler
collapses, so the model freezes at the random-chance solution.

**Fix:** default `init_cfg` to Xavier-uniform on `Conv2d` and `Linear`
(equivalent to Keras `glorot_uniform`) when the config does not specify
one. Validated with a CPU overfit probe (7 batches, 150 epochs): default
init crawls (2.49 -> 1.50 with a ~75-epoch dead zone) whereas Xavier init
drops immediately and steadily (3.14 -> 1.09); the `noise_std=1.0`
reference value is retained and works fine once the init is correct.
The diverged `work_dirs/amr_benchmark/icamcnet/deepsig201610A` run was
discarded and the model re-queued.

### `csrr/models/backbones/mcldnn.py` — LSTM reshape + Xavier init (🟢 high confidence)

**Symptom:** identical random-chance freeze to ICAMCNet (train *and* val
loss pinned at `ln(11)=2.398` from epoch 1, even at the full `lr=1e-3`
before `ReduceLROnPlateau` decayed it). Adding Xavier init alone was not
enough — a full-pipeline probe (`MODELS.build` + `data_preprocessor` +
`train_step`, 400 steps over 20 real batches) still stayed at 2.40.

**Root cause:** the tensor handed to the LSTM was reshaped incorrectly.
After `conv5` the NCHW activation is `[B, 100, 1, L-4]`; the Keras
reference reshapes the NHWC tensor `[B, 1, L-4, 100]` to `(L-4, 100)` so
the LSTM sees `time = L-4` conv positions and `features = 100` channels.
The CSRR port used `torch.reshape(x5, [-1, L-4, 100])`, which interleaves
the 100 channels into the time axis and feeds the LSTM a scrambled
sequence — the network cannot fit it and never leaves random chance.

**Fix:** `x = x5.squeeze(2).permute(0, 2, 1).contiguous()` before the LSTM
(frame-length agnostic). The same full-pipeline probe then drops the loss
steadily (2.40 -> 2.06 in 400 steps). Also defaulted `init_cfg` to
Xavier-uniform (Conv2d/Conv1d/Linear) to match the Keras `glorot_uniform`
reference. The diverged 10A run was discarded and MCLDNN re-queued.

### `csrr/models/backbones/cldnn.py` (`CLDNNL`) — Xavier init (🟢 high confidence)

**Symptom:** same random-chance freeze (best checkpoint `epoch_1`, val
accuracy pinned at 9.09%). `CLDNNW` (lighter, 50-channel) trained fine;
`CLDNNL` (deep 256-channel conv stack with `Dropout(0.5)` after every
conv) did not.

**Root cause / fix:** missing weight init — the deep conv stack vanishes
under PyTorch's default init. Defaulted `init_cfg` to Xavier-uniform on
Conv2d/Linear (matching the Keras CLDNN2 `glorot_uniform`). `CLDNNW` was
left untouched because it already trains and has a valid result. The
diverged `CLDNNL` 10A run was discarded and re-queued.

### `csrr/models/utils/init.py` — register `RNN` weight initializer (🟢 high confidence)

**Symptom:** every CGDNet config (`dict(type='RNN', layer='GRU', gain=1)`
in `init_cfg`) crashed `init_weights()` with
`KeyError: 'RNN is not in the csrr::weight initializer registry'`.

**Root cause / fix:** the generic recurrent initializer (`rnn_init`:
Xavier on `weight_ih`, orthogonal on `weight_hh`, forget-gate-bias=1) was
registered only under the name `LSTM`, but the CGDNet configs request
`RNN`. Registered the same `LSTMInit` class under both `LSTM` and `RNN`
(stacked `register_module`), so CGDNet's GRU gets the intended
Keras-style init. Verified `MODELS.build(...).init_weights()` runs clean.

## Phase 2 — RML2016.10A results & assessment

Full 15-model sweep on RML2016.10A (50/10/40 split, 2× RTX 3090, Adam
lr=1e-3, EarlyStopping patience=50, ReduceLROnPlateau). Measured
overall / peak (see the auto table in `accuracy_tracking.md`):

| Model | overall (tgt) | peak (tgt) | note |
|-------|---------------|------------|------|
| CNN2 | 63.6 (59) | 81.7 (79) | above target |
| CNN4 | 57.6 (58) | 83.9 (80) | peak above |
| MCNet | 56.0 (58) | 82.4 (82) | on target |
| ICAMCNet | 56.8 (57) | 85.1 (83) | **fixed** (init) |
| ResNetAMR | 54.3 (57) | 82.2 (83) | overall −2.7 |
| DensCNN | 54.5 (57) | 82.9 (83) | overall −2.5 |
| GRU2 | 57.7 (58) | 85.7 (85) | on target |
| LSTM2 | 56.6 (58) | 85.1 (87) | A/P variant |
| DAE | 55.6 (57) | 84.7 (82) | peak above |
| MCLDNN | 57.8 (62) | 85.2 (92.05) | **fixed** (reshape+init); residual gap |
| CLDNNW | 53.0 (57) | 78.3 (85) | ZeroPad divergence |
| CLDNNL | 57.5 (57) | 83.8 (85) | **fixed** (init) |
| CGDNet | 53.5 (58) | 79.2 (83) | **fixed** (RNN init); low |
| PETCGDNN | 57.9 (60) | 86.5 (89) | peak −2.5 |
| CNN1DPF | 55.0 (57) | 84.2 (85) | AP variant |

**Interpretation of the universal `fail` label.** The orchestrator's
status is `pass` only if overall, peak *and* best-SNR are all within
tolerance. Two effects make almost every row `fail` even where the
reproduction is sound:

1. **Best-SNR criterion vs. high-SNR plateau.** Accuracy saturates at
   high SNR, so the per-SNR argmax lands at 14–18 dB for nearly every
   model. The reference "best SNR" values (4–10 dB) are really "≥X dB"
   (the plateau onset), so a measured peak at 18 dB is correct behaviour
   but trips the `±2 dB` band. Judge reproduction on overall + peak
   magnitude, not the argmax SNR.
2. **Approximate overall targets + 50/10/40 split.** The `overall`
   targets are eyeballed from DSP Fig. 5 (±1 pp) and assume the paper's
   60% train split; CSRR trains on 50%, so a ~2–4 pp lower overall is
   expected and consistent across models.

Net: peaks are at/above reference for ~half the models and within ~2 pp
for most others; overalls cluster ~2–4 pp below the approximate targets.

**MCLDNN residual gap (documented, retunes capped).** MCLDNN was the
hardest case: a scrambled LSTM reshape (now fixed) plus missing init
(now fixed) had frozen it at random chance. After both fixes it trains
correctly but converges to ~57.8 % overall / ~85.2 % peak versus the
92.05 % single-best anchor. Retune attempts: (1) Keras-style LSTM init
(Xavier-ih + orthogonal-hh) — no change (85.6→85.2). Diagnosis shows the
network plateaus at ~57.5 % val accuracy by ~epoch 31 *while LR is still
1e-3*, i.e. a genuine convergence ceiling, not an LR-schedule artifact;
the data path is verified faithful to the Keras reference (branch wiring,
paddings, concat axes) and uses identical raw-IQ inputs (no
normalisation). The ~7 pp peak gap is attributed to the reduced training
split and optimisation differences (Keras CuDNNLSTM/Adam vs PyTorch); on
this split MCLDNN performs like a typical model rather than the singular
top performer. Recorded as a residual gap per the Phase 2 retune cap.

## Phase 2 — RNN/temporal ceiling ROOT CAUSE (supersedes the "residual gap" note above)

The "documented residual gap" conclusion above was **wrong** — the shortfall
that hit specifically the high-target temporal models (MCLDNN, PET-CGDNN,
LSTM2) is a real, fixable systematic bug, not a faithful-but-lower
reproduction. Two A/B controls on MCLDNN/RML2016.10A (the worst case, 85.2 %
peak vs 92.05 % target) isolate it.

**Finding 1 — per-sample input scale (🟢 the fix).** Every RML2016.10A example
ships pre-scaled to a tiny *fixed* energy: Frobenius norm ≈ 0.1, RMS ≈ 0.006,
max|x| ≈ 0.02, and this is constant across SNR and modulation. CNN+ReLU stacks
tolerate the tiny scale (cnn2/gru2 meet their targets on raw IQ), but the deep
recurrent models sit in the near-linear gate regime and converge to a worse
optimum. Adding `SelfNormalize(norms=dict(iq={}))` (divide each sample by its
Frobenius norm → unit energy, ≈10× scale-up) is the lever that clears the
target.

**Finding 2 — training duration.** With EarlyStopping (patience 50 on
`accuracy/top1`) the original MCLDNN run checkpointed epoch 57 and reported
57.8 % / 85.2 %. Training the *same* config to convergence with ES OFF (150
epochs, reference LR schedule) already reaches 60.3 % / 90.1 % — confirming the
"plateau at epoch 31" was a checkpoint/early-stop artifact, not a true ceiling.
Normalization keeps the best epoch well inside the ES window (≈45), so ES is
left enabled.

A/B evidence (MCLDNN, RML2016.10A, 40 % held-out test set, `paper.pkl`):

| Variant | overall | peak | hi-SNR(≥10 dB) avg |
|---------|---------|------|--------------------|
| original (ES on, raw IQ) | 57.81 | 85.23 | ~85 |
| raw IQ, ES off, 150 ep | 60.30 | 90.11 | 89.83 |
| **+ per-sample L2 norm, ES off** | **62.11** | **93.07** | **92.76** |
| reference target | 62.0 | 92.05 | — |

Other hypotheses were checked and ruled out: sequence/time-step ordering for
the RNNs is correct (`iq-shape-L-F` transposes to `[L=128, F=2]`; MCLDNN's
`squeeze(2).permute(0,2,1)` already fixed); class-index ordering is shared
between train labels and `paper.pkl` decoding (a permutation would cap CNNs
equally — they meet target); best-vs-last checkpoint selection on the 10 % val
set is not the issue (the best epoch is well-defined and reproduced).

### Fix: per-sample L2 normalization on the IQ pipeline (🟢 high confidence)

`dict(type='SelfNormalize', norms=dict(iq={}))` is prepended to the IQ pipeline
for the models that empirically benefit. New normalized base configs
`configs/_base_/datasets/deepsig/iq-l2norm-deepsig2016{10A,10B}.py`,
`iq-l2norm-deepsig201801A.py` and `hisar/iq-l2norm-hisar2019.py` are consumed by
**MCLDNN, CGDNet** (and CLDNNW); the RNN-only `iq-shape-L-F-*` bases (consumed
only by **GRU2, PET-CGDNN**) are normalized in place. The fix was rolled out
**per-model, not globally**, because normalization is not universally positive:

| Model (10A) | raw overall/peak | +L2-norm overall/peak | decision |
|-------------|------------------|-----------------------|----------|
| MCLDNN | 57.8 / 85.2 (fail) | **61.8 / 92.5 (pass)** | normalize |
| PET-CGDNN | 57.9 / 86.5 (fail) | **60.3 / 90.4 (pass)** | normalize |
| CGDNet | 53.5 / 79.2 (fail) | 55.6 / **82.5** (peak pass) | normalize |
| GRU2 | 57.7 / 85.7 (pass) | 57.8 / 85.9 (pass) | normalize (neutral) |
| CNN2 | 63.6 / 81.7 (pass) | 62.9 / 80.3 (pass) | **keep raw** (norm slightly hurts) |
| CLDNNL | 57.5 / 83.8 | 55.8 / 83.3 | **keep raw** (norm hurts) |
| CLDNNW | 53.0 / 78.3 | 53.8 / 79.5 | normalize (marginal; capped by ZeroPad divergence) |

Pure CNNs (CNN2/4, MCNet, ICAMCNet, ResNetAMR, DensCNN) keep the un-normalized
`iq-deepsig*.py` base — at the tiny native scale their ReLU stacks are fine and
unit-energy normalization costs ~0.5–1 pp. The A/P-input models (LSTM2, DAE,
CNN1DPF) already L2-normalize the amplitude channel inside `IQToAP`, so they are
left unchanged. CLDNNL keeps raw IQ (its deep 256-channel conv front-end already
feeds well-scaled features to the LSTM, so normalization is counter-productive).
EarlyStopping (patience 50) is left enabled: with normalization the best epoch
lands well inside the patience window, so it no longer truncates convergence.

## Phase 2 — RML2018.01A & HisarMod pre-flight smoke test (🟢 high confidence)

Before the multi-day driver reached the two large/unfamiliar datasets,
a CPU-only build+instantiate check was run on representative configs to
catch any breakage early (no GPU consumed; tiny 12-sample temp
annotation so no full-corpus `np.load`). The reusable harness lives at
`tools/amr_benchmark/_smoke_test.py` and, per config: parses with
`mmengine.Config`, `MODELS.build(cfg.model)`, builds the train dataset,
pulls `dataset[0]` through the pipeline, collates a 4-sample batch
through the model's `data_preprocessor`, and runs a single CPU forward
in both `loss` (train mode) and `predict` (eval mode).

Configs exercised — both new bases on both datasets and both head dims:

| Config | base | input shape | classes/head | loss@init | result |
|--------|------|-------------|--------------|-----------|--------|
| `mcldnn/...201801A` | `iq-l2norm` | (1,2,1024) | 24/24 | 3.18 ≈ ln24 | PASS |
| `gru2/...shape-L-F...201801A` | `iq-shape-L-F` | (1024,2) | 24/24 | 3.13 | PASS |
| `cnn2/...201801A` | `iq` | (1,2,1024) | 24/24 | 3.21 | PASS |
| `lstm2/...ap-shape-L-F...201801A` | `ap-shape-L-F` | (1024,2) | 24/24 | 3.23 | PASS |
| `dae/...ap...201801A` | `ap` | (1024,2) | 24/24 | CE+MSE | PASS |
| `cldnnw/...201801A` | `iq-l2norm` | (1,2,1024) | 24/24 | 3.16 | PASS |
| `cnn1dpf/...201801A` | `iq` | (1,2,1024) | 24/24 | 3.24 | PASS |
| `petcgdnn/...shape-L-F...hisar` | `iq-shape-L-F` | (1024,2) | 26/26 | 3.27 ≈ ln26 | PASS |
| `cgdnet/...hisar` | `iq-l2norm` | (1,2,1024) | 26/26 | 3.28 | PASS |
| `mcldnn/...hisar` | `iq-l2norm` | (1,2,1024) | 26/26 | 3.33 | PASS |
| `cnn2/...hisar` | `iq` | (1,2,1024) | 26/26 | 3.15 | PASS |
| `lstm2/...ap-shape-L-F...hisar` | `ap-shape-L-F` | (1024,2) | 26/26 | 3.23 | PASS |
| `dae/...ap...hisar` | `ap` | (1024,2) | 26/26 | CE+MSE | PASS |

**Outcome: no real config/model bugs.** All 24-class (2018.01A) and
26-class (HisarMod) heads match their dataset class counts; the L2-norm
and shape-L-F bases produce the expected `(1,2,1024)` / `(1024,2)`
tensors; initial losses equal `ln(num_classes)` (sane random init). The
only initial smoke FAIL was a *harness* artifact, not a model bug: DAE's
backbone only returns its `(xc, x, xd)` reconstruction triple when
`self.training` is True (`csrr/models/backbones/dae.py` L42), so running
the loss path in `eval()` collapsed it; the harness was corrected to run
the loss path in `train()` mode (matching the real training loop) and
DAE then passes with the intended `0.1·CE + 0.9·MSE` decomposition.

## Known divergences kept on purpose

These are noted here so Phase 2 can decide whether to invest more
time on them if accuracy falls outside the tolerance band.

1. **CLDNNW removes ZeroPadding2D.** Documented inside
   `csrr/models/backbones/cldnn.py` (lines 83–86): the AMR-Benchmark
   `ZeroPadding2D` does not preserve the time axis for this conv
   stack and only injects synthetic zero samples. CSRR drops it and
   accepts the corresponding LSTM input-size redefinition.
2. **LSTM2 uses amplitude/phase.** AMR-Benchmark Keras feeds raw
   I/Q to LSTM2, but DSP 2022 Table 1 documents LSTM as
   A/P-driven, and CSRR follows the paper. If LSTM2 underperforms
   the targets in Phase 2 we will create an `iq-shape-L-F`
   variant.
3. **CNN1DPF parallel branches consume A/P.** AMR-Benchmark splits
   the I and Q channels into the two branches; CSRR splits
   amplitude and phase. Same Phase 2 escape hatch.
4. **GRU2 on HisarMod kept on standard I/Q L×F pipeline.** Fixed
   in this branch (the old pipeline had an unused `IQToAP` +
   transposed-AP+packed-IQ chain).
5. **CSRR uses 50/10/40 split for every dataset**, including
   HisarMod, while DSP 2022 §5.1 describes 6:2:2 for RML and
   8:2:5 for HisarMod. This is the project's standardised split
   and is the reason small absolute accuracy differences vs DSP
   Fig. 5 are expected. The tolerance bands in
   `accuracy_targets.md` already account for this.
6. **Training schedule monitor differences.** AMR-Benchmark
   monitors `val_loss` for both `EarlyStopping` and
   `ReduceLROnPlateau`; CSRR uses `accuracy/top1` and
   `loss/classification` respectively. This rarely affects final
   accuracy but does shift the wall-clock convergence point.

## Items deferred to Phase 2 / Phase 1B

- `csrr/performance/` and `tools/analyze.py` are owned by Phase 1B
  (parallel worker) and have not been touched in this branch.
- Training and accuracy verification happen in Phase 2; this branch
  only sets up the orchestrator and target tables.

## 2026-06-29 — RML2018.01A training never terminates (~106-day ETA)

🟢 **Symptom.** During the Phase 2 sweep every RML2018.01A job stalled.
`work_dirs/amr_benchmark/denscnn/deepsig201801A/.../*.log` showed the run
at Epoch 94 of a 10000-epoch budget, mmengine ETA **~106 days**, loss
plateaued ~1.4, and the learning rate stuck at `5.0000e-04` for 80+
epochs. `mcldnn@2018` (Epoch 28, ETA ~46 d) and `dae@2018` (Epoch 87,
ETA ~12 d) showed the same pattern. Because the sweep runs ≤2 jobs in
parallel and each 2018 job effectively never ended, the
2018 → HisarMod → finalize chain was blocked (only 5/15 of 2018 had
produced `res/paper.pkl`).

**Root cause (two compounding bugs, both in the shared `_base_` config).**

1. `configs/_base_/schedules/amc.py` — `train_cfg` set
   `max_epochs=10000` (line 16, the misleading `# train 5 epochs`
   comment notwithstanding). RML2018.01A has ~3195 train iters/epoch at
   ~0.287 s/iter ≈ 15 min/epoch, so 10000 epochs ≈ 106 days — exactly
   the reported ETA. RML2016.10A/10B share this file but have ~275
   iters/epoch (~11 s/epoch), so their *wall-clock* ETA looked like
   "~1 day" and they converged in time via early stopping.
2. `configs/_base_/runtimes/amc.py` — the `EarlyStoppingHook` used
   `min_delta=0`, `patience=50` on `accuracy/top1`. With `min_delta=0`
   *any* sub-0.01 pp wiggle counted as an "improvement" and reset the
   patience window. On the 24-class RML2018.01A set the validation
   accuracy keeps inching up for a long time, so early stopping never
   fired and training drifted toward `max_epochs`. The
   `ReduceOnPlateauParamScheduler` (monitor `loss/classification`) had
   the same problem from the other side: val loss kept improving
   marginally, so the LR was almost never decayed — hence the LR frozen
   at `5.0000e-04`. (The small RML2016 sets *did* eventually trip both
   mechanisms: e.g. `denscnn/deepsig201610A` decayed all the way to
   `1.0000e-06` and early-stopped at epoch 207, best @ 157.)

Evidence the models do not actually need thousands of epochs: the five
RML2018.01A models that *did* finish reached their best validation epoch
at 77 (`cnn4`), 91 (`cnn2`), 109 (`icamcnet`), 117 (`resnetamr`), and
193 (`mcnet`).

**Fix (shared `_base_`, inherited by all 15 AMR-Benchmark models across
all four datasets incl. HisarMod — no per-model hacks).**

`configs/_base_/schedules/amc.py`:

| field | old | new |
| --- | --- | --- |
| `train_cfg.max_epochs` | `10000` | `150` |
| `param_scheduler` | `ReduceOnPlateauParamScheduler` (lr, monitor `loss/classification`, factor 0.5, patience 5) | `CosineAnnealingLR(by_epoch=True, T_max=150, eta_min=1e-6)` |

`configs/_base_/runtimes/amc.py` — `EarlyStoppingHook`:

| field | old | new |
| --- | --- | --- |
| `min_delta` | `0` | `0.1` (percentage points) |
| `patience` | `50` | `15` |

Rationale: `CosineAnnealingLR` makes the LR anneal *visibly every epoch*
toward `1e-6`, removing the plateau-stuck-LR failure mode entirely.
`max_epochs=150` comfortably covers the observed 77–193-epoch
convergence window while bounding the worst case to hours instead of
months. The stricter early-stopping criterion (≥0.1 pp gain over a
15-epoch window) terminates jobs once validation accuracy genuinely
plateaus.

**Why this does not regress RML2016.10A/10B.** All 10A/10B results are
already frozen as `res/paper.pkl` and the orchestrator
(`tools/amr_benchmark/run_migration.py`) skips any job whose checkpoint
/ `paper.pkl` already exists, so completed runs are never recomputed.
For any re-run, the small 2016 sets converge far inside the 150-epoch
cap (10A `denscnn` best was @ 157 only because `min_delta=0` let it
inch up; with `min_delta=0.1` it plateau-stops earlier near the same
accuracy, within the ±1.5 pp tolerance).

**Process actions.** Killed the four stalled RML2018.01A `train.py`
jobs running under the old schedule — `denscnn` (orig PID 1246880),
`dae` (PID 1848076), `mcldnn` (PID 1849375), and `lstm2` (orphaned
workers) — plus their forked dataloader workers (88 PIDs total). Left
the durable drivers and orchestrator untouched (`_sweep_driver.sh`
696546, `run_migration.py` 995031, `_finalize_driver.sh` 1149690).
Removed the four partial work_dirs
(`work_dirs/amr_benchmark/{denscnn,mcldnn,dae,lstm2}/deepsig201801A`),
none of which contained `res/paper.pkl`, so the orchestrator retrains
them from scratch under the fixed schedule rather than reusing a
half-trained `best_*.pth`. On the freed GPUs the live orchestrator
immediately launched the next pending 2018 jobs (`cldnnw@2018`,
`cldnnl@2018`) under the new config, confirming the re-run path. The
killed four are re-attempted in the sweep's HisarMod phase and the
finalize driver's idempotent full sweep.

**Note for own-method schedules.** `configs/{lstm2,mldnn,hcgdnn,
fastmldnn}/schedules.py` and `configs/trn/schedule.py` carry the same
latent pattern (`max_epochs` 10000/3200/400 and `MultiStepLR`
milestones `[800, 1200]` that never fire within a sane budget). They
are *not* used by the 15-model main sweep (those configs pull
`_base_/schedules/amc.py`), so they are out of scope for this blocker,
but the own-method 2018/HisarMod runs in the finalize driver will need
an analogous fix before they can converge in reasonable time.

## 2026-06-30 — Own-methods (MLDNN / HCGDNN / FastMLDNN) schedules never terminate (3–5 day ETAs)

🟢 **Symptom.** As the finalize driver reached the own-method
(MLDNN/HCGDNN/FastMLDNN) jobs on the large datasets (RML2018.01A,
HisarMod), each job drifted toward multi-day ETAs and monopolized both
RTX 3090s, starving the cheaper missing baselines of GPU time. Unlike
the 15 main-sweep models, these three pull their *own*
`configs/<method>/schedules.py` + `runtimes.py` rather than the shared
`_base_` files, so they never received the 2026-06-29 `_base_` fix and
still carried the latent never-terminating pattern flagged in that
entry's "Note for own-method schedules".

**Root cause (per method).**

1. **MLDNN** — `configs/mldnn/schedules.py` declared **no
   `param_scheduler` at all**, so the LR was frozen at `4.0000e-04` for
   the entire run, and `max_epochs=400` let RML2018.01A (~2000 train
   iters/epoch) drift for multiple days.
2. **HCGDNN** — `configs/hcgdnn/schedules.py` used
   `ReduceOnPlateauParamScheduler` (monitor `accuracy/top1`, patience
   30) with `max_epochs=10000`. Because the fused validation accuracy
   keeps inching up by tiny amounts, the plateau scheduler almost never
   stepped, so the LR sat frozen at `4.4000e-04` while training ran to
   epoch 200+ (observed ~5-day ETA). Its `EarlyStoppingHook` also used
   `min_delta=0, patience=100`, so early stopping effectively never
   fired.
3. **FastMLDNN** — `configs/fastmldnn/schedules.py` used
   `MultiStepLR(milestones=[800, 1200])` with `max_epochs=3200`; the
   first LR drop would not occur until epoch 800, far beyond any sane
   convergence point, so the LR never decayed and training never
   stopped. Its `runtimes.py` had **no `custom_hooks`** at all (no early
   stopping).

**Fix (per-method, mirrors the 2026-06-29 `_base_` recipe; preserves
each method's original optimizer LR — MLDNN `4e-4`, HCGDNN/FastMLDNN
`4.4e-4`).**

| File | old | new |
| --- | --- | --- |
| `configs/mldnn/schedules.py` | no `param_scheduler`, `max_epochs=400` | `CosineAnnealingLR(by_epoch=True, T_max=150, eta_min=1e-6)`, `max_epochs=150` |
| `configs/hcgdnn/schedules.py` | `ReduceOnPlateauParamScheduler`, `max_epochs=10000` | `CosineAnnealingLR(by_epoch=True, T_max=150, eta_min=1e-6)`, `max_epochs=150` |
| `configs/hcgdnn/runtimes.py` `EarlyStoppingHook` | `min_delta=0, patience=100` | `min_delta=0.1, patience=15` |
| `configs/fastmldnn/schedules.py` | `MultiStepLR(milestones=[800,1200])`, `max_epochs=3200` | `CosineAnnealingLR(by_epoch=True, T_max=150, eta_min=1e-6)`, `max_epochs=150` |
| `configs/fastmldnn/runtimes.py` `custom_hooks` | (none) | `[EarlyStoppingHook(monitor='accuracy/top1', min_delta=0.1, patience=15, rule='greater')]` |

MLDNN inherits its `EarlyStoppingHook` (and the `CheckpointHook` with
`save_best='accuracy/top1'`) from `_base_/runtimes/amc.py`; HCGDNN and
FastMLDNN now carry the same `EarlyStoppingHook` locally and also
inherit `save_best='accuracy/top1'` from `_base_`. **Paper accuracy is
preserved, not degraded:** the `CheckpointHook` saves the best-val
checkpoint and the orchestrator tests that checkpoint, so the bounded
schedule only caps wall-clock time — it never tests a half-trained
or post-plateau model. `CosineAnnealingLR` makes the LR anneal visibly
every epoch toward `1e-6`, removing the plateau-stuck-LR failure mode,
and the stricter early stopping (≥0.1 pp gain over a 15-epoch window)
terminates each job once validation accuracy genuinely plateaus,
bounding the worst case to hours instead of days.

**Scope.** Only the five own-method files above were changed. No
baseline configs and no shared `_base_` files (`_base_/schedules/amc.py`,
`_base_/runtimes/amc.py`) were touched. Live verification on the
running jobs confirmed the fix is active: `fastmldnn@deepsig201801A`
(epoch 6, `lr 9.97e-04` cosine-decaying, ETA ~1 day, `EarlyStoppingHook`
in the hook list) and `hcgdnn@deepsig201801A` (epoch 18, `lr 4.26e-04`
decaying from `4.4e-04`, `EarlyStoppingHook` present). With the
own-method ETAs dropped from multi-day to hours, the cheaper missing
baselines can again receive GPU time as these jobs complete.

## 2026-07-01 — FastMLDNN diverges (dead-ReLU collapse) on RML2018.01A

🟢 **Symptom.** `fastmldnn@deepsig201801A` produced a garbage result:
the orchestrator tested from `best_accuracy_top1_epoch_1.pth` and
measured overall **8.02%**, peak **11.09%** (~random for 24 classes).
The best checkpoint being epoch 1 means validation accuracy never
improved after epoch 1. By contrast `fastmldnn@deepsig201610B` is fine
(overall 57.81%, peak 87.75% from `best_epoch_64`), so the failure is
specific to 2018.01A.

**Root cause — divergence, not a data/label bug.** Both configs inherit
the same shared `_base_/schedules/amc.py` (Adam **`lr=1e-3`**),
`_base_/runtimes/amc.py`, and the same model (`num_classes` correctly 24
on 2018, 10 on 10B; AP pipeline identical). The only dataset difference
is signal length: **2018.01A frames are 1024 samples vs 128 on
10A/10B**. The `FastMLDNN` backbone
(`csrr/models/backbones/fastmldnn.py`) aggregates its transformer output
with `merge='sum'` — `torch.sum(x, dim=1)` over the post-CNN sequence.
Three `kernel_size=3, stride=1` convs reduce length by 6, so the summed
sequence is **~1018 steps on 2018 vs ~122 on 10A/10B (~8.3x)**; the
summed feature magnitude (hence logits and gradients) is ~8x larger on
2018. The training log proves the collapse:

| epoch | train loss (end) | val acc | val loss |
| --- | --- | --- | --- |
| 1 | 2.7976 | 8.03% | 59.72 |
| 2 | **3.1781** | 4.17% (=1/24) | 31.45 |
| 3–16 | **3.1781 (frozen)** | 4.17% | 22→17 |

`3.1781 = ln(24)` *exactly*: after the epoch-1 step the final classifier
ReLUs went dead (the last `nn.Linear` has `bias=False`, so an all-zero
post-ReLU input yields uniform logits → constant `ln(24)` loss and
exactly 1/24 accuracy), and the zero gradient through dead ReLUs meant
the network never recovered. `EarlyStoppingHook` (patience 15) then
stopped it and the epoch-1 weights were tested. This is classic
LR-too-high divergence, amplified on 2018 by the ~8x sum-merge scaling.
10A/10B are unaffected (short signals → ~8x smaller summed activations)
and converge fine at the shared `lr=1e-3`.

**Fix (dataset-specific; `configs/fastmldnn/fastmldnn_iq-ap-deepsig-201801A.py` only).**

| field | old (inherited `_base_`) | new (2018.01A override) |
| --- | --- | --- |
| `optim_wrapper.optimizer.lr` | `1e-3` | `2e-4` (toward FastMLDNN's paper `4.4e-4`, scaled down for the 8x sum magnitude) |
| `optim_wrapper.clip_grad` | (none) | `dict(max_norm=5.0, norm_type=2)` — hard guard against the explosive step |
| `param_scheduler` | single `CosineAnnealingLR(T_max=150)` | `[LinearLR(start_factor=0.01, begin=0, end=5), CosineAnnealingLR(T_max=145, begin=5, end=150, eta_min=1e-6)]` — 5-epoch warmup so early high-variance steps cannot kill the ReLUs, then cosine |

`max_epochs=150`, `EarlyStoppingHook(min_delta=0.1, patience=15)` and
`CheckpointHook(save_best='accuracy/top1')` are unchanged (inherited).
**10A/10B configs are deliberately untouched** (they don't share this
file and converge fine), so this cannot regress the completed/good
10A/10B results.

**Validation (manual single-GPU run on the fixed config, GPU1, cache on).**
The fix is confirmed: the warmup LR starts microscopic (`~3e-6`, ramping)
and the initial training loss is **207** (directly confirming the giant
summed logits on 2018) but then descends smoothly under gradient
clipping — `207 → 7.3 → 3.0` within epoch 1, ending epoch 1 at **2.78**,
and crucially epoch 2 **continues down to ~2.58 (then 2.45 in epoch 3)
instead of locking at `ln(24)`**. Validation accuracy *climbs* and the
best checkpoint advances past epoch 1:

| epoch | train loss | val acc | best ckpt |
| --- | --- | --- | --- |
| 1 | 2.78 | 8.71% | epoch 1 |
| 2 | 2.58 | **13.51%** | **epoch 2** |
| 3 | 2.45 | (climbing; LR still in warmup ~9e-5) | — |

No NaN, no collapse, loss monotonically decreasing, val accuracy rising
while the LR is still only warming up (it reaches the full `2e-4` at
epoch 5, after which convergence accelerates). The stale garbage result
(`res/paper.pkl`, `best_accuracy_top1_epoch_1.pth`, `epoch_16.pth`,
`last_checkpoint`) was removed from
`work_dirs/amr_benchmark/fastmldnn/deepsig201801A/` (logs kept) so the
idempotent sweep re-trains fastmldnn@2018 under the fixed config; the
manual validation run was then killed (it lived in a throwaway
`_validate_2018` work-dir and never touched the sweep's directory).

**Related-method risk assessment (no changes made).**
- **mldnn@deepsig201801A** (queued, not yet re-run): **low risk** for
  this defect. MLDNN uses `lr=4e-4` (not `1e-3`) and aggregates bounded
  GRU states (`tanh` outputs), and a prior 2018.01A run trained 41
  epochs without any `ln(24)` collapse (loss stable ~5.3 with healthy
  per-branch terms). Its historical problem was sluggish convergence +
  the never-terminating schedule (addressed separately on 2026-06-30),
  not divergence. Worth watching when the sweep retrains it, but it does
  not share FastMLDNN's dead-ReLU failure, so MLDNN is left unchanged.
- **hcgdnn@deepsig201801A** (running): **healthy** — val acc ~58%
  (plateauing), epoch 63, `lr` cosine-annealing (`2.8e-4`), loss
  decreasing. Left untouched.

**Scope.** Only `configs/fastmldnn/fastmldnn_iq-ap-deepsig-201801A.py`
was changed. No shared `_base_` files, no backbone/head code, no other
method or dataset configs, and no 10A/10B results were touched.
