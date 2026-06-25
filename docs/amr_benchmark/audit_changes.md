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
