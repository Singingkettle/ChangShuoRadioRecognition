# Own Methods — Paper-Exact Alignment (MLDNN / FastMLDNN / HCGDNN)

**Date:** 2026-07-14  
**Policy:** Tier A — `campaign_mode: paper_exact`. Data split **50/10/40 (5:1:4) is paper-native** for these three; do **not** attribute residual gaps to “TF 6:2:2 concession.”  
**Architecture freeze:** topology frozen; retune hyperparameters / init / schedule / multi-loss weights / documented pipeline only.  
**Related:** [`goal_mode.md`](./goal_mode.md) · [`goals.json`](../../configs/amr_benchmark/retune/goals.json) · [`fastmldnn_paper_comparison.md`](./fastmldnn_paper_comparison.md) · [`own_methods_results.md`](./own_methods_results.md)

Paper PDF (Windows path historically unavailable). Authoritative recipe sources in-repo:

| Method | Paper configs |
|--------|---------------|
| FastMLDNN | `configs/fastmldnn/paper/` (+ `fastmldnn_stage2_iq-ap-deepsig-201610A.py`) |
| MLDNN | `configs/mldnn/original/` |
| HCGDNN | `configs/hcgdnn/original/` |

---

## 1. Dataset coverage (paper-target vs measured-only)

| Method | RML2016.10A | RML2016.10B | RML2018.01A | HisarMod |
|--------|-------------|-------------|-------------|----------|
| **MLDNN** | **paper-target** (~62 / ~92) | measured-only | measured-only | measured-only |
| **FastMLDNN** | **paper-target** (63.24 / 92) | measured-only | measured-only | measured-only |
| **HCGDNN** | **paper-target** (64.9 / 93) | measured-only | measured-only | measured-only |

Only **10A** has siege/campaign paper-exact stop criteria. Other three datasets remain measured-only (no fail conversion required by paper numbers).

---

## 2. Live measured vs paper (10A) — honest status

| Method | Paper overall / peak | Measured | Tracking (−1.5/−1.0) | Paper-exact | Notes |
|--------|----------------------|----------|----------------------|-------------|-------|
| **MLDNN** | 62.0 / 92.0 | **62.31 / 92.73** | **pass** | **pass** | Meets fig-read targets; keep config frozen |
| **FastMLDNN** | 63.24 / 92.0 | **61.02 / 91.52** | **fail** (61.02 < 61.74) | **fail** (−2.22 / −0.48) | Synced retune best `esoff300` (2026-07-14) |
| **HCGDNN** | 64.9 / 93.0 | **63.04 / 93.11** | **fail** (63.04 < 63.40) | **fail** (−1.86 overall; peak OK) | Prior lr/ES retunes **hurt**; need paper schedule |

---

## 3. Hyperparameter tables — paper vs CSRR AMR

### 3.1 FastMLDNN @ RML2016.10A

| Knob | Paper `paper/fastmldnn_iq-ap-deepsig201610A.py` | Paper channel full (`…-channel-…`) | CSRR AMR default | Retune best (`esoff300`) |
|------|-----------------------------------------------|------------------------------------|------------------|--------------------------|
| **LR** | Adam **4.4e-4** | Adam **1.054e-4** | 4.4e-4 | 4.4e-4 |
| **Schedule** | MultiStep `[800,1200]` γ=0.3 (w/ max 400 → **never drops**) | MultiStep `[20,80,400,600,760]` γ=0.3 | **Cosine** T_max=150 | Cosine T_max=300 |
| **Epochs** | **400** | **3200** | **150** | **300** |
| **ES** | none (old runner) | none | top1, Δ0.1, pat **15** | **off** |
| **Loss** | CE only (this file) | **Focal** + balance **0.5** | CE, **β=0** | CE + **β=0.5** |
| **Dropout** | default backbone | **dp=0.07** + pretrained | dp=0.5 | **dp=0.07** |
| **Init** | TruncNormal head | Pretrained backbone + TruncNormal | default / none | Xavier Conv1d + TruncNormal |
| **Batch** | **80** | **640** | **640** | 640 |
| **IQ L2** | none | none | none | **SelfNormalize** (not paper) |
| **Split** | train+val ann historically in `paper/` | same | **50/10/40** (paper-native policy) | same |

**Primary gaps vs paper-exact:** default **β=0**; Cosine+tight ES vs paper MultiStep/long budget; prior best still used non-paper **IQ L2**. Next siege: paper MultiStep + β without L2 (`siege_fastmldnn_10a_paper.json`).

### 3.2 MLDNN @ RML2016.10A

| Knob | Paper `original/` | CSRR AMR |
|------|-------------------|----------|
| **LR** | Adam **4e-4** | **4e-4** |
| **Schedule** | **fixed** | Cosine T_max=150 |
| **Epochs** | **400** | **150** + ES |
| **ES** | none | top1 Δ0.1 pat 15 (`amc.py`) |
| **Loss** | 4-head multi-loss (snr/low/high/merge) | same weights=1 each |
| **Dropout** | **0.5** | **0.5** |
| **Init** | (legacy default) | Xavier Conv2d |
| **Batch** | **80** | (dataset base; typically larger) |
| **Arch** | BiGRU + SAFN + gradient truncation | **matched** |

**Status:** Already **paper-exact pass** under fig-read 62/92. No siege priority unless targets revised from PDF.

### 3.3 HCGDNN @ RML2016.10A

| Knob | Paper `original/` | CSRR AMR |
|------|-------------------|----------|
| **LR** | Adam **4.4e-4** | **4.4e-4** |
| **Schedule** | MultiStep `[800]` γ=0.3 | **Cosine** T_max=150 |
| **Epochs** | total_epochs **1600** | **150** + ES |
| **ES** | none | top1 Δ0.1 pat **15** + `HCGDNNHook` |
| **Loss** | CE × heads (CNN/BiGRU1/BiGRU2) | same |
| **Batch** | **640** | **640** |
| **Init** | default | default |
| **Fusion** | val-time Optimization merge | `HCGDNNWeightsAccuracy` + persistent fusion weights |

**Primary gaps:** Cosine+early stop vs paper MultiStep@800 / 1600-ep budget. Prior retunes (`lr1e3`, `es_patience25`) **regressed**. Peak already ≥ 93; overall −1.86 pp is a **training-recipe** problem under paper-exact (not split blame).

---

## 4. Retune plan (paper-exact siege firepower)

### Priority order

| Rank | Pair | Action |
|------|------|--------|
| **P0** | FastMLDNN × 10A | Paper-recipe siege: MultiStep γ=0.3 + β=0.5 + dp=0.07; variants w/ lr=4.4e-4 / 1.054e-4 / fixed-LR 400ep; **no L2**; stop only at ≥63.24 / ≥92 |
| **P1** | HCGDNN × 10A | After FastMLDNN: restore paper MultiStep milestone 800 (or scaled) + longer budget, ES off/relaxed; **do not** raise lr to 1e-3 again |
| **P2** | MLDNN × 10A | **Hold** — already pass; document only |

### Tier B (deprioritized)

CLDNNW@2018, ICAMCNet Hisar ES loops, ResNetAMR ports, etc. → `campaign_mode: approximate` only. Remaining small gaps may cite 5:1:4 vs TF 6:2:2.

---

## 5. Measured-only snapshots (no paper bar)

| Method | 10B overall/peak | 2018 overall/peak | Hisar overall/peak |
|--------|------------------|-------------------|--------------------|
| MLDNN | 65.06 / 93.62 | 57.94 / 90.77 | 60.06 / 73.63 |
| HCGDNN | 65.04 / 93.71 | 58.72 / 93.52 | 57.39 / 70.16 |
| FastMLDNN | 57.81 / 87.75 | 48.05 / 77.45 | **5.98 / 7.90** (broken — needs recipe audit) |

FastMLDNN@Hisar is a separate failure mode (not paper-target); defer until 10A paper-exact closes.

---

## 6. Wave-12 author-exact strategy (2026-07-27)

**Why:** waves 1–11 saturated — FastMLDNN variants cluster 59.8–61.02 (paper 63.24),
HCGDNN 62.7–63.31 (paper 64.9). Every wave used *ported* schedules
(Cosine + tight ES, or shortened MultiStep). `git diff HEAD origin/main` shows the
local working copy **replaced the author's published training recipes** (commit
e9c3c99 "baseline recipe"); the paper's official release (origin/main, linked from
the TCCN abstract) uses different ones. Wave-12 restores them verbatim.

### Recovered author recipes (origin/main)

| Method | Author recipe (published) | Never tried in waves 1–11 |
|--------|---------------------------|----------------------------|
| FastMLDNN stage-1 | beta=0, dp=0.5, Adam 4.4e-4, MultiStep[800,1200] γ0.3, **3200 ep, no ES**, batch 640, no L2 | full budget + no-ES (author's best-val hit ep 648) |
| FastMLDNN stage-2 | **Pretrained stage-1 backbone** → dp=**0.07**, beta=**0.5**, **constant lr 1.054e-4**, up to 3200 ep | the entire two-stage pipeline |
| FastMLDNN paper/ per-dataset | **batch 80**, constant 4.4e-4 (milestones beyond max_epochs=400), CE only, TruncNormal head | batch 80 (all waves used 640) |
| HCGDNN | Adam 4.4e-4 + **ReduceOnPlateau**(top1, f0.3, patience 30, min 1e-7), **ES min_delta=0 patience=100**, HCGDNNHook, batch 640 | adaptive plateau schedule + patient ES |

### Init / dropout audit (user-requested)

- **HCGDNN backbone** already does per-layer init matching best practice: Xavier-uniform
  Conv2d, Xavier ih / **orthogonal hh** per GRU gate chunk, zero biases (`csrr/models/backbones/hcgdnn.py`).
  No init gap; the gap is the schedule.
- **FastMLDNN**: `dp` drives CNN dropout **and** the transformer-encoder dropout;
  head dropout is fixed 0.5. The paper uses dp=0.5 from scratch and dp=**0.07 only
  in stage-2 with a pretrained backbone** — running dp=0.07 from scratch (our wave-2/3
  best) under-regularises the transformer, which is consistent with the 61.02 ceiling.
  Author init: TruncNormal(std 0.02) Linear head; backbone default (+ Pretrained in stage-2).

### Wave-12 runs

| Variant | Where | Config |
|---------|-------|--------|
| `author_stage1_ms3200_w12` | H100 GPU0 | `wave12_fastmldnn_..._author_stage1_ms3200.py` |
| `author_stage2_from_esoff300best_w12` | H100 GPU2 | author stage-2 recipe warm-started from historical best 61.02 |
| `author_plateau_es100_w12` + `author_plateau_ft_from_exact800_w12` | H100 (queued after FastMLDNN pair) | HCGDNN author plateau recipe (scratch + FT from 63.30) |
| `author_iqap_b80_fixedlr400_w12` | local GPU1 | batch-80 constant-LR paper per-dataset recipe |

**Follow-up (manual/steward):** when `author_stage1_ms3200_w12` finishes, launch the
author stage-2 fine-tune from its best-val checkpoint (create config with `load_from`
pointing at `author_stage1_ms3200_w12/best_accuracy_top1_epoch_*.pth`).

### Infra root-cause fixed alongside

CUDA default device ordering (FASTEST_FIRST) mapped `CUDA_VISIBLE_DEVICES=0` to
physical GPU 1 on the H100 box, so nvidia-smi-based idle detection and CUDA-based
job placement disagreed — jobs packed onto busy GPUs while others idled. All
launchers now export `CUDA_DEVICE_ORDER=PCI_BUS_ID` (`gpu_pool_keepalive.sh`,
`gpu_keepalive.sh`, `retune_sweep.py`, `run_migration.py`), and
`retune_model_siege.py` now skips queue entries in status `running` so a second
fill-idle-GPUs orchestrator cannot re-claim an in-flight entry.

## 7. Config source map

| Artifact | Path |
|----------|------|
| Goals / tiers | `configs/amr_benchmark/retune/goals.json` |
| FastMLDNN paper siege | `configs/amr_benchmark/retune/siege_fastmldnn_10a_paper.json` |
| Tracking | `docs/amr_benchmark/accuracy_tracking.md` |
| FastMLDNN narrative compare | `docs/amr_benchmark/fastmldnn_paper_comparison.md` |
