# TensorFlow AMR-Benchmark ↔ CSRR PyTorch Alignment Audit

**Date:** 2026-07-14  
**Scope:** Audit + documentation only (architecture freeze). Optional bug-fix list is deferred unless a clear TF mismatch remains.  
**TF reference:** [`Richardzhangxx/AMR-Benchmark`](https://github.com/Richardzhangxx/AMR-Benchmark) cloned at `/tmp/AMR-Benchmark`  
**TF commit:** `6a129c411b73e521fc3f04ccb8e1e58f8af71eb1`  
**Live tracking:** 23 `pass` / 38 `fail` / 11 `measured` (`accuracy_tracking.md` orchestrator table, 2026-07-14 FastMLDNN sync).  
**Constraint:** CSRR **train/val/test = 50/10/40 (5:1:4)** is intentional. Do **not** revert the default split to 6:2:2; quantify impact instead.

Related docs: [`accuracy_targets.md`](./accuracy_targets.md), [`audit_changes.md`](./audit_changes.md), [`accuracy_tracking.md`](./accuracy_tracking.md), [`goal_mode.md`](./goal_mode.md), [`retune_campaign.md`](./retune_campaign.md), [`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md), [`fastmldnn_paper_comparison.md`](./fastmldnn_paper_comparison.md), [`retune_results.md`](./retune_results.md).

---

## 0.0 Correction — Own methods vs TF split narrative (2026-07-14)

**MLDNN / FastMLDNN / HCGDNN are Tier A own methods.** Their papers already use
**5:1:4 (50/10/40)** as the native protocol under the project policy. Earlier
sections of this audit (and some retune notes) that explain FastMLDNN/HCGDNN
residuals as “TF 6:2:2 vs CSRR 5:1:4” are **wrong for these three** — that
framing applies only to **Tier B** TensorFlow AMR-Benchmark ports.

| Tier | Models | Split story | Campaign success |
|------|--------|-------------|------------------|
| **A** | MLDNN, FastMLDNN, HCGDNN | 5:1:4 = **paper-native** (not a TF concession) | **paper_exact** |
| **B** | All other AMR ports | 5:1:4 vs TF 6:2:2 may explain small gaps | **approximate** (−1.5 / −1.0 pp) |

See [`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md) and
[`goal_mode.md`](./goal_mode.md). FastMLDNN/HCGDNN gaps must be closed (or
waived) via **paper recipe alignment** (schedule, multi-loss β, init, epochs,
ES), not by blaming the split.

---

## 0. Executive summary (中文友好)

**为什么 campaign 几乎不动了（分层说明）：**

1. **Tier B — Split 天花板（TF ports only）** — TF/DSP 在 RML 用 **6:2:2**；我们固定 **50/10/40**。对 **TF 移植模型** overall 可系统性低 ~2–4 pp。在 approximate 模式下小差距可接受。**不适用于** MLDNN/FastMLDNN/HCGDNN。
2. **Schedule 漂移** — TF：`Adam(1e-3)` + `ReduceLROnPlateau` + 长 ES；CSRR 默认 Cosine+紧 ES。对 Tier A 应对照 **各自 paper configs**，不是 TF。
3. **真实结构分叉（CLDNNW，Tier B）** — ZeroPad 删除 → LSTM 维差；超参追不上 paper。
4. **Tier A near-miss** — FastMLDNN@10A **61.02 / 91.52** vs paper **63.24 / 92**（−2.22 / −0.48）：归因于 **β/调度/epoch 与 paper 不一致**，**不是** 5:1:4 vs TF。HCGDNN 63.04 vs 64.9：同理（paper MultiStep/长预算未对齐）。MLDNN@10A 已 **paper-exact pass**。
5. **Tracking FastMLDNN 已同步（2026-07-14）** — 勿用 39% 当当前最好；现为 61.02/91.52，仍 tracking+paper-exact fail。

**结论：** Tier A 全力 paper-exact；Tier B 以 approximate 收口。全矩阵统一 paper-exact **不做**。

---

## 1. Shared training / data protocol matrix

| Dimension | TF AMR-Benchmark (RML201610a typical) | CSRR (AMR sweep default) | Impact |
|-----------|----------------------------------------|---------------------------|--------|
| **Data split** | Per-(mod,snr) **600/200/200 → 6:2:2**; seed `2016` | Global JSON **50/10/40** (`train.json` / `validation.json` / `test.json`) | **~2–4 pp overall**; intentional |
| **Hisar split** | Official Train/Test mats; Train → **0.8/0.2** train/val; Test = held-out | Same 50/10/40 of unified corpus | Larger Hisar overall gaps |
| **Optimizer** | `optimizer='adam'` (Keras default lr **1e-3**, β1/β2 0.9/0.999) | Adam **lr=1e-3** (`configs/_base_/schedules/amc.py`) | Matched for baselines; own-methods use 4e-4 / 4.4e-4 |
| **Weight decay** | None explicit | None default | Matched |
| **Batch** | **400** | **400** (FastMLDNN **640**) | Mostly matched |
| **LR schedule** | `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patince=5→default 10, min_lr≈1e-6/1e-7)` | **`CosineAnnealingLR(T_max=150, eta_min=1e-6)`** | Material for fine convergence |
| **Early stopping** | `monitor='val_loss', patience=50` (Hisar some 20–30) | `monitor='accuracy/top1', min_delta=0.1, patience=15` | Truncates slow models (FastMLDNN) |
| **Epoch budget** | 1000–10000 + ES | **150** hard cap + ES | Intentional wall-clock fix |
| **Loss** | Typically categorical CE; DAE multi-loss 0.1·CE+0.9·MSE | Same CE / DAE weights | FastMLDNN tracking path had `beta=0` (paper 0.5) |
| **Checkpoint** | `ModelCheckpoint(monitor='val_loss')` | `save_best='accuracy/top1'` | Rarely flips ranking |
| **Input scale (IQ)** | Raw native RML scale (tiny energy ~0.1) | Per-model: **SelfNormalize** for MCLDNN/CGDNet/CLDNNW/GRU/PET; raw for CNNs | Critical for temporal models |
| **Input scale (AP)** | `to_amp_phase` + L2 on amp channel | `IQToAP` L2 on amp | Matched for LSTM2/DAE/CNN1DPF |

### TF split code (RML 6:2:2)

```25:26:/tmp/AMR-Benchmark/RML201610a/MCLDNN/dataset2016.py
            train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
```

### TF Hisar (Train 80/20 + separate Test)

```83:95:/tmp/AMR-Benchmark/HisarMod/CNN1/main.py
n_examples = train.shape[0]
n_train = int(n_examples * 0.8)
n_val = int(n_examples * 0.2)
...
X_test = test
Y_test = test_labels
```

### CSRR schedule / ES

```15:26:configs/_base_/schedules/amc.py
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=150,
    eta_min=1e-6,
)
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
```

```23:23:configs/_base_/runtimes/amc.py
custom_hooks = [dict(type='EarlyStoppingHook', monitor='accuracy/top1', min_delta=0.1, patience=15, rule='greater')]
```

---

## 2. Split analysis (critical)

### 2.1 Protocols

| Protocol | Train | Val | Test | Where |
|----------|-------|-----|------|-------|
| **CSRR (frozen)** | **50%** | **10%** | **40%** | All AMR + own-method configs |
| **TF / DSP RML** | **60%** | **20%** | **20%** | `dataset2016.py` / `rmldataset2016.py` |
| **TF Hisar** | ≈0.8×Train | ≈0.2×Train | full official Test | `HisarMod/*/main.py` |
| **Own-paper FastMLDNN/MLDNN legacy** | `train_and_validation.json` | often = test | test | `configs/*/paper/`, `original/` — **not** AMR default |

### 2.2 Expected systematic bias

| Effect | Rough magnitude | Rationale |
|--------|-----------------|-----------|
| Train fraction 0.50 vs 0.60 | **−2 to −4 pp overall** on RML | Fewer labels; Phase-2 + DSP footnote in `accuracy_targets.md` / `audit_changes.md` |
| Test 0.40 vs 0.20 | Slightly **lower variance**, similar mean | Larger test does not raise mean accuracy |
| Val 0.10 vs 0.20 | Noisier early-stop / best-ckpt | Can clip late gains under strict ES |
| Hisar 50/10/40 vs official Train/Test + 8:2 | **−3 to −8+ pp** possible | Different corpus partitioning; MCNet/PET/ResNet Hisar fails |

**Why paper-exact campaign is hard under 5:1:4:**  
Many targets sit near Fig. 5 eyeballs. A faithful PyTorch port under 50% train that lands **2 pp low** fails both tracking (±1.5) and paper-exact. FastMLDNN residual **63.24 − 61.02 ≈ 2.2 pp** is *inside* the split band. HCGDNN **64.9 − 63.04 ≈ 1.86** likewise.

### 2.3 Recommendations (do not change default)

**(a) Dual reporting (preferred):** keep 5:1:4 as the published CSRR protocol; report  
`split_adjusted_target ≈ paper_target − 2.5` (±1 pp band) for RML overall; mark Hisar separately.

**(b) Optional calibration A/B (config-only, one model):** re-train **MCLDNN@10A** (already pass control) and **FastMLDNN@10A** once with TF-matching 6:2:2 annotations (or temporary `ann_file` that mimics 600/200/200). **Do not** merge into default. Purpose: measure Δ overall attributable to split alone.

---

## 3. Control models (headline passes)

| Dimension | MCLDNN (pass 10A/10B/2018) | LSTM2 (pass 10B/2018) | PETCGDNN (pass 10A/10B) |
|-----------|----------------------------|------------------------|-------------------------|
| **Input** | Raw I/Q multi-branch | **A/P** (TF *and* CSRR) | Raw I/Q + PET rotation |
| **Norm** | CSRR: `SelfNormalize` IQ; TF: raw | Amp L2 in TF `norm_pad_zeros` / CSRR `IQToAP` | CSRR L2 on L×F IQ |
| **Init** | Xavier (+ LSTM init) | Default / OK | OK after Q-rotation fix |
| **Arch** | Reshape fixed to Keras `(L-4,100)` | 2×LSTM(128) | Rotation sign fixed vs TF |
| **10A overall** | 61.75 vs ~62 | 56.57 vs ~58 (peak short) | 60.26 vs ~60 |

### Correction vs older docs

Older notes claimed “TF LSTM2 = raw I/Q, CSRR = A/P.” **False for this repo’s TF code.** LSTM2 `rmldataset2016.py` calls `to_amp_phase` then `norm_pad_zeros` on the amplitude channel. Retune `lstm2/.../iq_input` collapsed to **14.95%**, confirming A/P is required.

Similarly, **1DCNN-PF TF also uses A/P** (`to_amp_phase` then `X[:,:,0/1]` branches). CSRR `ap-deepsig*` is modality-aligned; prior “AP vs IQ” divergence note was incorrect for TF mainline.

---

## 4. Systematic divergence matrix (failing / near-pass focus)

Status from orchestrator table unless noted. Retune bests from `retune_results.md`.

### 4.1 Cross-cutting (all models)

| Dimension | TF | CSRR | Notes |
|-----------|----|------|-------|
| Split | 6:2:2 / Hisar official | **50/10/40** | Primary systematic bias |
| Opt / lr / wd / batch | Adam 1e-3 / 400 | Same for `_base_` | |
| LR schedule | Plateau on **val_loss** | **Cosine 150** | |
| Early stop | **val_loss**, patience 50 | **acc/top1**, Δ0.1, patience 15 | |
| Weight init | Often `glorot_uniform` | Xavier when coded; **CLDNNW missing** | |

### 4.2 Per-model (selected)

| Model | Status highlight | Input | Norm | Init | Layer / pad notes | Primary residual cause |
|-------|------------------|-------|------|------|-------------------|------------------------|
| **FastMLDNN@10A** | Tracking **39.32** stale; retune **61.02/91.52** | I/Q+A/P | L2 in best stack | Xavier+TruncNormal | Arch frozen; `beta=0` in default config | Split (~2pp) + protocol; stack exhausted |
| **HCGDNN@10A** | 63.04 vs 64.9 | I/Q | — | — | Own method | Within split noise; lr retunes *hurt* |
| **ICAMCNet@Hisar** | peak **98.56** (retune ~98.5) vs 100 | I/Q | raw | Xavier | GaussianNoise σ=1 | Peak ceil; ES loops futile |
| **CLDNNW@2018** | **37.19/53.33** (retune best 43.8/65.7) | I/Q | L2 | missing Xavier | **ZeroPad removed**; LSTM dim −32 | **Structural** + long-seq |
| **CGDNet@2018** | **35.87/51.67** (retune best 49.6/75.9) | I/Q | L2 | RNN init fixed | Size formula matches TF 4056 | Opt/schedule/long-seq; not pad |
| **CLDNNL@2018** | 46.67/81.40 | I/Q | raw (by design) | Xavier | Deep 256-stack | Long-seq + ES |
| **CNN2@2018** | 42.35/65.23 | I/Q | raw | — | Matched arch | Severe 2018 underfit vs siblings |
| **MCNet@Hisar** | 56/79 vs 70/97 | I/Q | raw | — | Param-parity port | Hisar split + poor Hisar CNN in paper |
| **CNN1DPF@10A/B** | marginal fails | **A/P both sides** | amp L2 | — | Parallel Conv1D | Split + schedule; modality OK |
| **LSTM2@10A** | peak 85.1 vs 87 | A/P | amp L2 | — | Matched | Peak-only; iq_input retune failed |
| **ResNet/Dens/CLDNN\*** | many −2–3 pp overall | I/Q | mostly raw | mixed | Mostly matched | Split band |

FastMLDNN / HCGDNN are **not** in TF AMR-Benchmark; compare to **paper** configs under the same 50/10/40 constraint.

---

## 5. Root causes for stuck near-pass / no campaign progress

### 5.1 FastMLDNN@10A — 61.02 vs paper 63.24 (Tier A)

| Fact | Evidence |
|------|----------|
| Tracking row **39.32%** is **pre-siege baseline** (`beta=0`, ES patience 15) | `accuracy_tracking.md` FASTMLDNN 10A |
| Best siege: `beta05_xavier_l2_dp007_esoff300` → **61.02 / 91.52** | `retune_results.md` 2026-07-11 |
| Default AMR config still `beta=0` | `configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py` L17 |
| Residual **~2.2 pp** | **Paper recipe** (MultiStep/β/epochs), **not** TF split — see §0.0 |
| Peak **91.52 vs ~92** | Still short of paper-exact peak |

**Interpretation:** Closed bug-like gap (39→61). Remaining gap is **paper-training alignment** under architecture freeze. Next: `siege_fastmldnn_10a_paper.json`.

### 5.2 HCGDNN@10A — 63.04 vs 64.9 (Tier A)

- Peak already **93.11 ≥ 93**. Overall shortfall **1.86 pp** — **paper schedule** gap (Cosine+ES vs MultiStep@800 / 1600 ep), not TF split.  
- Retunes (`lr1e3`, `es_patience25`) **regressed** overall.  
- Keep **paper_exact**; restore paper MultiStep after FastMLDNN paper siege.

### 5.3 ICAMCNet@Hisar — peak ~98.5 vs 100

- Overall **82** already above ~80 target; fail driven by peak **−1.44 pp** and/or best-SNR.  
- Many `--force` ES retunes stuck **98.45–98.56**.  
- TF uses same GaussianNoise(1)+glorot; peak 100% on official Hisar partition is an extreme ceiling.  
- **Waive paper-exact peak** or require dual-protocol Hisar; do not loop ES.

### 5.4 CLDNNW / CGDNet @2018 — catastrophic

| | CLDNNW | CGDNet |
|-|--------|--------|
| Sweep | 37.2 / 53.3 | 35.9 / 51.7 |
| Best retune | 43.8 / 65.7 | 49.6 / 75.9 |
| TF target | ~55 / ~88 | ~57 / ~92 |
| Structural | **ZeroPad gap (dim −32)** | Size OK (4056=4056) |

CLDNNW gap is **architecture-policy** (freeze forbids restoring ZeroPad without explicit policy change). CGDNet remains an **optimization / long-seq** problem under cosine+tight ES; further gains need schedule paradigm closer to TF plateau, not more lr knobs alone.

### 5.5 Tracking FastMLDNN (synced 2026-07-14)

- **Was stale:** auto-table **39.32%** vs siege **61.02 / 91.52**.  
- **Now:** `paper.pkl` + best ckpt synced into `work_dirs/amr_benchmark/fastmldnn/deepsig201610A` (`PROMOTE_NOTE.txt`); auto-table rebuilt via `run_migration.py --skip-train --skip-test`.  
- **Classify:** paper-exact **fail**; tracking **fail** (61.02 < 61.74); split-adjusted triage (~60.74) would clear — secondary only.  
- Default config still `beta=0` (architecture/config freeze).

---

## 6. Concrete TF ↔ CSRR code diffs (file:line)

### Diff A — Data split 6:2:2 vs 50/10/40

**TF** (`/tmp/AMR-Benchmark/RML201610a/MCLDNN/dataset2016.py` L25–26): per-cell 600/200/(rest→200).  
**CSRR** (`configs/_base_/datasets/deepsig/iq-deepsig201610A.py`): `ann_file='train.json'|'validation.json'|'test.json'` (50/10/40).  

→ Dominant accuracy offset; **intentional**.

### Diff B — CLDNNW ZeroPadding removed (structural)

**TF** — pad before each of three convs:

```23:44:/tmp/AMR-Benchmark/RML201610a/CLDNN/rmlmodels/CLDNNLikeModel.py
    input_x_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_x)
    layer11 = Conv2D(50, (1, 8), padding='valid', ...)(input_x_padding)
    ...
    layer11_padding = ZeroPadding2D((0, 2), ...)(layer11)
    ...
    layer12 = ZeroPadding2D((0, 2), ...)(layer12)
    layer13 = Conv2D(50, (1, 8), ...)(layer12)
    concat = keras.layers.concatenate([layer11, layer13])
    ...
    lstm_out = CuDNNLSTM(units=50)(concat)
```

**CSRR** — pads deleted; LSTM `input_size=(L*2-28)*2` (456 @ L=128 vs TF ≈488):

```93:110:csrr/models/backbones/cldnn.py
        # Compared to AMR-Benchmark, we remove the Padding layer.
        ...
        self.lstm = nn.LSTM(input_size=(self.frame_length * 2 - 7 * 4) * 2, hidden_size=50, batch_first=True)
```

→ Explains persistent CLDNNW underperformance; restoring pad would be an **architecture change** (policy exception).

### Diff C — MCLDNN LSTM reshape (control — fixed)

**TF:**

```43:45:/tmp/AMR-Benchmark/RML201610a/MCLDNN/rmlmodels/MCLDNN.py
    x= Reshape(target_shape=((124,100)),name='reshape')(x)
    x = CuDNNLSTM(units=128,return_sequences = True)(x)
    x = CuDNNLSTM(units=128)(x)
```

**CSRR (aligned):**

```84:91:csrr/models/backbones/mcldnn.py
        # x5 is NCHW [B, 100, 1, L-4]. The Keras reference reshapes ...
        x = x5.squeeze(2).permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
```

→ After +L2, MCLDNN passes on RML — gold control that parity + norm works under 5:1:4 for *some* models.

### Bonus — PET rotation (control — fixed)

TF: `I'=I cos+Q sin`, `Q'=Q cos−I sin` (`PETCGDNN.py` L41–44).  
CSRR: same formula (`petcgdnn.py` L27–33). Passes on 10A/10B.

---

## 7. Top 5 alignment retunes (ROI under 5:1:4, architecture frozen)

Ranked for **paper-exact or tracking promotion**, not for rewriting nets:

| Rank | Retune | Why high ROI | Expectation under 5:1:4 |
|------|--------|--------------|-------------------------|
| **1** | **Promote FastMLDNN `beta=0.5` + L2 + dp=0.07 + ES-off/long cosine** into default AMR config | Closes 39→61; tracking honesty | Still ~**−2 pp** vs 63.24 unless split A/B |
| **2** | **Adopt split-adjusted targets / dual reporting** (ops, not GPU) | Stops futile siege on HCGDNN/ICAMC peak | Makes many “fails” into protocol-aware pass |
| **3** | **One-shot 6:2:2 calibration** on MCLDNN + FastMLDNN (config-only) | Quantify Δ(split) in pp | Do not change default 5:1:4 |
| **4** | **CLDNNW: policy decision on ZeroPad** (only if chasing 2018 paper) | Hyperparams exhausted (~+7 pp max) | Without pad restore, 2018 paper-exact unlikely |
| **5** | **Restore TF-like Plateau(val_loss)+patience≥50 for CGDNet/CLDNN\*@2018 only** | Cosine+tight ES mismatch | May recover mid-teens pp; still long jobs |

**Deprioritize:** more ICAMCNet Hisar ES loops; HCGDNN lr sweeps; LSTM2 raw-IQ; FastMLDNN lr micro-sweeps past 300 ep (diminishing).

---

## 8. Optional clear bugs vs TF (architecture freeze)

| Item | Verdict |
|------|---------|
| MCLDNN reshape / ICAMC Xavier / PET sign / Hisar L=1024 | **Already fixed** |
| CLDNNW ZeroPad | **Intentional divergence**, not an accidental bug |
| LSTM2 / CNN1DPF “IQ vs AP” | **Docs were wrong**; code already matches TF A/P |
| FastMLDNN default `beta=0` | Config omission vs **paper**, not vs TF repo |
| New code change this audit | **None** — no clear accidental bug remaining that is both TF-mismatched and freeze-safe |

---

## 9. Pass/fail picture under intentional 5:1:4

```
Orchestrator (2026-07-08): 23 pass / 38 fail / 11 measured
```

Rough taxonomy of the 38 fails:

| Bucket | Examples | Action |
|--------|----------|--------|
| **Split-band (−1.5…−4 pp overall)** | Dens/ResNet/CNN1DPF/HCGDNN/MCNet-10A | Adjust targets or tolerate |
| **Peak ceiling / SNR argmax** | ICAMC Hisar, Dens peak@SNR | Soften peak/SNR criteria |
| **Stale tracked low + retune higher** | FastMLDNN 39 vs 61 | Promote stack; still short paper by ~2 pp |
| **Structural ZeroPad** | CLDNNW all sets, esp. 2018 | Policy call |
| **Long-seq collapse** | CGDNet/CLDNNW/CLDNNL/CNN2@2018 | Schedule A/B; not more same knobs |
| **Hisar partition** | MCNet/PET/GRU/LSTM Hisar | Dual Hisar protocol note |

---

## 10. References

- TF clone: `/tmp/AMR-Benchmark` @ `6a129c4`  
- DSP 2022 survey (Zhang et al.) Fig. 5 / §5.1 splits  
- CSRR: `docs/amr_benchmark/*`, `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py`  
- Retune log: `docs/amr_benchmark/retune_results.md`
)
