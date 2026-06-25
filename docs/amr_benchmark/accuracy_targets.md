# AMR-Benchmark Reference Targets

This document collects the **reference architecture, training
hyperparameters and reported accuracy** for every (model × dataset)
pair we plan to reproduce. All numbers are taken either from the
AMR-Benchmark Keras source code (`https://github.com/Richardzhangxx/AMR-Benchmark`,
folders `RML201610a/`, `RML201610b/`, `RML2018/`, `HisarMod/`) or
from the DSP 2022 survey paper "Deep Learning Based Automatic
Modulation Recognition: Models, Datasets, and Challenges" (Zhang et
al., 2022) — specifically Fig. 5 (per-SNR accuracy curves) and
Table 4 (model size / minimum validation loss). Cached copy:
`/home/citybuster/.cursor/projects/home-citybuster-Projects-ChangShuoRadioRecognition/agent-tools/edf9ef8f-cafe-46ed-914f-15fdc3348b2d.txt`.

The peak accuracies in the DSP paper are quoted **only for the
single best-performing model per dataset** (MCLDNN, LSTM, LSTM,
CNN1/DenseNet); other models' peak values are read off Fig. 5
within ~±1 percentage point. For the orchestrator we therefore
adopt the following tolerance bands (per plan):

| Metric | Tolerance |
|--------|-----------|
| Overall test accuracy (mean over SNRs in `[-14, 18]` for DeepSig, `[-20, 18]` for HisarMod) | **±1.5 pp** vs reference |
| Peak accuracy at best SNR | **±1.0 pp** vs reference |
| Best SNR (dB) | **±2 dB** vs reference |

## Model ↔ CSRR backbone mapping

| AMR-Benchmark name | CSRR backbone class | Input modality (paper) | Status |
|--------------------|---------------------|------------------------|--------|
| CNN1 | `CNN2` | I/Q | matched |
| CNN2 (Multipath) | `CNN4` | I/Q | matched (kernels fixed to (2,8) in this branch) |
| MCNET | `MCNet` | I/Q | implemented in this branch |
| IC-AMCNet | `ICAMCNet` | I/Q | implemented in this branch |
| ResNet | `ResNetAMR` | I/Q | implemented in this branch |
| DenseNet | `DensCNN` | I/Q | matched |
| GRU | `GRU2` | I/Q (Keras code; A/P per Table 1) | matched (DeepSig); HisarMod base needed reshape fix |
| LSTM | `LSTM2` | A/P (per Table 1, CSRR convention) | matched |
| DAE | `DAE` | A/P + reconstruction | matched (DAEHead reinstated in this branch) |
| MCLDNN | `MCLDNN` | I/Q multi-branch | matched |
| CLDNN (West) | `CLDNNW` | I/Q | structural variant — padding intentionally simplified in CSRR |
| CLDNN2 | `CLDNNL` | I/Q | matched |
| CGDNet | `CGDNet` | I/Q | matched (frame_length fixed on long-seq configs) |
| PET-CGDNN | `PETCGDNN` | I/Q + rotation | matched (Q-rotation sign fixed in this branch) |
| 1DCNN-PF | `CNN1DPF` | I/Q parallel branches | matched (CSRR uses AP branches — see audit_changes.md) |

## Shared AMR-Benchmark training defaults

- Optimizer: Adam, `lr=0.001`, β1=0.9, β2=0.999
- Batch size: 400
- LR scheduler: `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr ∈ {1e-6, 1e-7})`
  (note: the AMR-Benchmark `patince=5` is a typo in every script and falls back to the Keras default of 10)
- EarlyStopping: `monitor='val_loss', patience=50` (HisarMod variants use 20–30 for ResNet/CLDNN/MCLDNN)
- Train/val/test split: 6:2:2 for RML datasets, 8:2:5 for HisarMod (per DSP §5.1).
  **CSRR uses 50/10/40 across all datasets** (the project's standardized
  split) — small absolute accuracy differences vs Fig. 5 are expected.

## Per-model targets

Conventions: "Arch key params" lists the most distinctive
hyperparameters (filter counts, kernels, dropout). "Ref peak" is
the highest per-SNR accuracy and the SNR at which it occurs, read
from DSP 2022 Fig. 5 unless noted. "Ref overall" is the mean
accuracy across the dataset's SNR range as plotted; for models
where the paper does not call out a number this is an approximate
band derived from Fig. 5 (±1 pp).

### CNN1 → `CNN2`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 2 convs (50 × 1×8) + Dropout 0.5 + Dense(256) + Dense(11) | epochs=10000, batch=400 | ~58–60% | ~78–80% @ ≥6 dB |
| RML2016.10B | same, classes=10 | epochs=10000 | ~62–65% | ~85% @ ≥4 dB |
| RML2018.01A | same, frame_length=1024 | epochs=10000 | ~58% | ~92% @ ≥18 dB |
| HisarMod | same, classes=26 | epochs=10000 | ~75% | **~100% @ ≥10 dB** (DSP §5.2 callout) |

Source: `RML201610a/CNN1/rmlmodels/CNN2Model.py` + `main.py`.

### CNN2 (Multipath) → `CNN4`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 4 convs (256/128/64/64 × 2×8) + Dropout 0.5 + Dense(128) + Dense(11) | epochs=1000, batch=400 | ~57–59% | ~80% @ ≥4 dB |
| RML2016.10B | classes=10 | epochs=1000 | ~62–64% | ~84% @ ≥2 dB |
| RML2018.01A | frame_length=1024 | epochs=1000 | ~55% | ~91% @ ≥18 dB |
| HisarMod | classes=26 | epochs=1000 | ~70% | ~98% @ ≥10 dB |

Source: `RML201610a/CNN2/rmlmodels/CNN2.py` + `main.py`.

### MCNET → `MCNet`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | stem (3×7)/64 + pre-block + 6 M-blocks + AvgPool(2,1) + Dense(11). 121.5k params (Table 4) | epochs=10000, batch=400, patience=50 | ~58% | ~82% @ ≥6 dB |
| RML2016.10B | AvgPool(2,1), classes=10. 121.2k params | epochs=1000 | ~62% | ~87% @ ≥4 dB |
| RML2018.01A | AvgPool(2,8), classes=24. 126.6k params | epochs=10000 | ~55% | ~92% @ ≥18 dB |
| HisarMod | AvgPool(2,8), classes=26. 127.4k params | epochs=10000 | ~70% | ~97% @ ≥10 dB |

Source: `RML201610a/MCNET/rmlmodels/MCNET.py` + `main.py`. The DSP
paper highlights MCNET's poor convergence on HisarMod (Table 4
val_loss 1.136 — highest among CNNs).

### IC-AMCNet → `ICAMCNet`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 4 (1×k) convs (64/64/128/128) + Dropout 0.4 + Dense(128) + GaussianNoise(σ=1) + Dense(11). 1.26M params | epochs=1000, batch=400 | ~57% | ~83% @ ≥6 dB |
| RML2016.10B | classes=10. 1.26M params | epochs=1000 | ~62% | ~87% @ ≥4 dB |
| RML2018.01A | frame_length=1024, classes=24. 8.61M params | epochs=1000 | ~58% | ~92% @ ≥18 dB |
| HisarMod | classes=26. 8.61M params | epochs=1000 | ~80% | **~100% @ ≥10 dB** |

Source: `RML201610a/IC-AMCNet/rmlmodels/ICAMC.py` + `main.py`.

### ResNet → `ResNetAMR`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | conv(256,1×3)/relu, conv(256,2×3), broadcast-add residual, conv(80,1×3)×2, Dropout 0.6, Dense(128), Dense(11). 3.10M params | epochs=10000, patience=50 | ~57% | ~83% @ ≥6 dB |
| RML2016.10B | classes=10. 3.10M params | epochs=1000 | ~62% | ~87% @ ≥4 dB |
| RML2018.01A | frame_length=1024, classes=24. 21.5M params | epochs=1000 | ~57% | ~91% @ ≥18 dB |
| HisarMod | classes=26. 21.5M params. patience=20 | epochs=1000 | ~80% | ~100% @ ≥10 dB |

Source: `RML201610a/ResNet/rmlmodels/ResNet.py` + `main.py`.

### DenseNet → `DensCNN`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | conv(256,1×3) + conv(256,2×3) + concat → conv(80,1×3) + concat → conv(80,1×3) + Dropout 0.6 + Dense(128) + Dense(11). 3.28M params | epochs=10000 | ~57% | ~83% @ ≥6 dB |
| RML2016.10B | classes=10. 3.28M params | epochs=10000 | ~62% | ~87% @ ≥4 dB |
| RML2018.01A | classes=24. 21.6M params | epochs=1000 | ~58% | ~92% @ ≥18 dB |
| HisarMod | classes=26. 21.6M params. patience=20 | epochs=1000 | ~80% | **~100% @ ≥10 dB** (DSP §5.2 callout) |

Source: `RML201610a/DenseNet/rmlmodels/DenseNet.py` + `main.py`.

### GRU → `GRU2`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 2× CuDNNGRU(128) on raw I/Q (128×2) + Dense(11). 151.2k params | epochs=10000 | ~58% | ~85% @ ≥4 dB |
| RML2016.10B | classes=10. 151.1k params | epochs=10000 | ~63% | ~91% @ ≥2 dB |
| RML2018.01A | input 1024×2, classes=24. 152.9k params | epochs=10000 | ~59% | ~95% @ ≥18 dB |
| HisarMod | classes=26. 153.1k params | epochs=10000 | ~73% | ~98% @ ≥10 dB |

Source: `RML201610a/GRU2/rmlmodels/GRUModel.py` + `main.py`.

### LSTM → `LSTM2`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 2× CuDNNLSTM(128) on raw I/Q (128×2) + Dense(11). 201.1k params | epochs=10000 | ~58% | ~87% @ ≥4 dB |
| RML2016.10B | classes=10. 201.0k params | epochs=1000 | ~64% | **~94% @ 18 dB** (DSP §5.2 callout) |
| RML2018.01A | input 1024×2, classes=24. 202.8k params | epochs=10000 | ~60% | **~98.39% @ 22 dB** (DSP §5.2 callout) |
| HisarMod | classes=26. 203.0k params. patience=30 | epochs=10000 | ~73% | ~98% @ ≥10 dB |

Source: `RML201610a/LSTM2/rmlmodels/CuDNNLSTMModel.py` + `main.py`.

> **Note:** AMR-Benchmark Keras code feeds raw I/Q to LSTM2 even
> though the DSP paper Table 1 lists LSTM as A/P-driven. CSRR
> currently uses the A/P-driven variant (project convention). This
> may shift LSTM accuracy by 1–3 pp on RML datasets; Phase 2 will
> re-train with both pipelines if numbers fall short of tolerance.

### DAE → `DAE`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 2× LSTM(32) on A/P (L×2) + Dense(32→16→11) + TimeDistributed Dense(2) decoder. Loss = 0.1·CE + 0.9·MSE. 1.06M params | epochs=10000 | ~57% | ~82% @ ≥6 dB |
| RML2016.10B | classes=10. 1.06M params | epochs=1000 | ~62% | ~85% @ ≥4 dB |
| RML2018.01A | classes=24. 67.1M params (huge dense decoder) | epochs=10000 | ~55% | ~90% @ ≥18 dB |
| HisarMod | classes=26. 67.1M params. DSP §5.4 notes DAE has severe confusion on HisarMod | epochs=10000 | ~40% | ~70% @ ≥10 dB |

Source: `RML201610a/DAE/rmlmodels/DAE.py` + `main.py`.

### MCLDNN → `MCLDNN`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | Multi-branch (full IQ, I, Q) Conv2D(50,2×8) + Conv1D(50,8)×2 → concat → Conv2D(50,1×8) → Conv2D(100,2×5) → 2× LSTM(128) → Dense(128,selu)×2 + Dropout 0.5 → Dense(11). 406.2k params | epochs=10000 | ~62% | **~92.05% @ 10 dB** (DSP §5.2 callout, top score for 10A) |
| RML2016.10B | classes=10. 406.1k params | epochs=10000 | ~65% | ~93% @ ≥4 dB |
| RML2018.01A | classes=24, L=1024. 407.9k params | epochs=10000 | ~60% | ~95% @ ≥18 dB |
| HisarMod | classes=26. 408.1k params. patience=30 | epochs=10000 | ~75% | ~99% @ ≥10 dB |

Source: `RML201610a/MCLDNN/rmlmodels/MCLDNN.py` + `main.py`.

### CLDNN (West) → `CLDNNW`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | ZeroPad → 3× Conv2D(50,1×8) with multipath skip → LSTM(50) → Dense(256, Dropout 0.5) + Dense(11). 164.4k params | epochs=10000 | ~57% | ~85% @ ≥6 dB |
| RML2016.10B | classes=10. 164.2k params | epochs=10000 | ~62% | ~89% @ ≥4 dB |
| RML2018.01A | classes=24, L=1024 (must override frame_length). 884.4k params | epochs=10000 | ~55% | ~88% @ ≥18 dB |
| HisarMod | classes=26, L=1024. 884.9k params | epochs=10000 | ~75% | ~98% @ ≥10 dB |

Source: `RML201610a/CLDNN/rmlmodels/CLDNNLikeModel.py` + `main.py`.

> **CSRR variant note:** the CSRR `CLDNNW` implementation
> intentionally removes the ZeroPadding2D layers used by the
> AMR-Benchmark reference (see in-code comment in
> `csrr/models/backbones/cldnn.py`). The padding does not preserve
> spatial size for this configuration so removing it avoids
> appending zero "phantom" samples to the time axis. Expected
> accuracy is comparable but may differ by ±1–2 pp.

### CLDNN2 → `CLDNNL`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | conv(256,1×3), conv(256,2×3), conv(80,1×3)×2 + Dropout 0.5 + reshape → LSTM(50) + Dense(128) + Dense(11). 517.6k params | epochs=10000 | ~57% | ~85% @ ≥4 dB |
| RML2016.10B | classes=10, Dropout 0.6. 517.5k params | epochs=10000 | ~62% | ~89% @ ≥2 dB |
| RML2018.01A | classes=24, L=1024 (override frame_length=1024 in this branch). 698.3k params | epochs=10000 | ~57% | ~92% @ ≥18 dB |
| HisarMod | classes=26, L=1024. 698.6k params. patience=30 | epochs=10000 | ~75% | ~98% @ ≥10 dB |

Source: `RML201610a/CLDNN2/rmlmodels/CLDNNLikeModel.py` + `main.py`.

### CGDNet → `CGDNet`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | 3× Conv2D(50,1×6) + GaussianDropout 0.2 + skip + GRU(50) + Dense(256) + Dense(11). 124.9k params | epochs=10000 | ~58% | ~83% @ ≥6 dB |
| RML2016.10B | classes=10. 124.7k params | epochs=1000 | ~62% | ~88% @ ≥4 dB |
| RML2018.01A | classes=24, L=1024 (override frame_length). 665.9k params | epochs=10000 | ~57% | ~92% @ ≥18 dB |
| HisarMod | not in AMR-Benchmark | N/A | (best-effort port) | (best-effort port) |

Source: `RML201610a/CGDNet/rmlmodels/CGDNN.py` + `main.py`. CSRR
adds an additional HisarMod config that the AMR-Benchmark does not
ship — treat it as exploratory.

### PET-CGDNN → `PETCGDNN`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | PET rotation (Dense(1) angle, sin/cos rotation) + Conv2D(75,8×2) + Conv2D(25,5×1) + GRU(128) + Dense(11). 71.9k params (smallest model in DSP Table 4) | epochs=10000 | ~60% | ~89% @ ≥6 dB |
| RML2016.10B | classes=10. 71.7k params | epochs=1000 | ~63% | ~92% @ ≥4 dB |
| RML2018.01A | classes=24, L=1024. 75.3k params | epochs=10000 | ~60% | ~95% @ ≥18 dB |
| HisarMod | classes=26. 75.6k params | epochs=10000 | ~75% | ~99% @ ≥10 dB |

Source: `RML201610a/PET-CGDNN/rmlmodels/PETCGDNN.py` + `main.py`.

### 1DCNN-PF → `CNN1DPF`

| Dataset | Arch key params | Hyperparams | Ref overall | Ref peak |
|---------|-----------------|-------------|-------------|----------|
| RML2016.10A | parallel I/Q Conv1D branches (4×64, k=3, Dropout 0.2) → concat → 5× Conv1D + MaxPool1D(2) → Dense(128, SELU)×2 + Dropout 0.5 + Dense(11) | epochs=10000 | ~57% | ~85% @ ≥6 dB |
| RML2016.10B | classes=10 | epochs=10000 | ~62% | ~88% @ ≥4 dB |
| RML2018.01A | classes=24, L=1024 | epochs=10000 | ~57% | ~91% @ ≥18 dB |
| HisarMod | not in AMR-Benchmark | N/A | (CSRR-only) | (CSRR-only) |

Source: `RML201610a/1DCNN-PF/rmlmodels/DCNNPF.py` + `main.py`.

> **CSRR variant note:** CSRR's `CNN1DPF` config feeds AP-derived
> channels (amplitude + phase) into the parallel branches whereas
> AMR-Benchmark splits raw I and Q. Both are valid; document the
> divergence and consider an I/Q config in Phase 2 if accuracy
> falls outside tolerance.

## DSP 2022 paper headline summary (Fig. 5)

| Dataset | Best model (paper) | Peak accuracy (paper) |
|---------|--------------------|------------------------|
| RML2016.10A | MCLDNN | **92.05%** @ 10 dB |
| RML2016.10B | LSTM | **94%** @ 18 dB |
| RML2018.01A | LSTM | **98.39%** @ 22 dB |
| HisarMod 2019.1 | CNN1, DenseNet | **≈100%** @ ≥10 dB |

These four cells are the strongest anchor points; any reproduction
that lands within ±1.5 pp of them is considered a pass.
