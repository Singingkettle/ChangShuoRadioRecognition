# AMR-Benchmark Retune Campaign

Continuous improvement plan to convert the **39 fail** entries in
[`accuracy_tracking.md`](./accuracy_tracking.md) to **pass** using hyperparameter
and initialization retunes (no backbone rewrites unless a bug is confirmed).

**Branch:** `feature/amr-benchmark-migration`  
**Orchestrator:** `tools/amr_benchmark/retune_sweep.py`  
**Baseline sweep:** `tools/amr_benchmark/run_migration.py`  
**Audit reference:** [`audit_changes.md`](./audit_changes.md)  
**TF alignment:** [`tf_alignment_audit.md`](./tf_alignment_audit.md)  
**Goal semantics:** [`goal_mode.md`](./goal_mode.md) · `configs/amr_benchmark/retune/goals.json`

---

## Executive note — priority reset (2026-07-14)

Live tracking after FastMLDNN sync: **23 pass / 38 fail / 11 measured**.

### Tiered campaign (see [`goal_mode.md`](./goal_mode.md))

| Tier | Models | Campaign stop | GPU priority |
|------|--------|---------------|--------------|
| **A — Own methods** | MLDNN, FastMLDNN, HCGDNN | **paper_exact** | **Siege firepower here first** |
| **B — TF ports** | everything else | **approximate** (−1.5 / −1.0) | Deprioritized |

Own-method **5:1:4 is paper-native** — do **not** treat FastMLDNN/HCGDNN gaps
as TF-split concessions. Alignment + plan:
[`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md).

### Why the counter looks stuck

1. **Stale tracking (fixed)** — FastMLDNN@10A synced to **61.02 / 91.52**; still
   tracking **fail** and paper-exact **fail** (honesty over cosmetics).
2. **Tier A recipe gap** — FastMLDNN residual ~2.2 pp and HCGDNN ~1.86 pp are
   **paper-schedule / multi-loss / epoch** mismatches under architecture freeze,
   not “wrong split.” MLDNN@10A already **paper-exact pass**.
3. **Tier B structural / futile** — CLDNNW ZeroPad freeze; ICAMCNet Hisar peak
   ~98.5 vs 100 exhausted/waived. Stop burning GPU on identical ES loops;
   approximate OK for remaining small TF-port gaps (5:1:4 vs TF 6:2:2 allowed).

**Next fire:** FastMLDNN@10A paper-recipe siege
(`siege_fastmldnn_10a_paper.json`), then HCGDNN paper MultiStep — not
CLDNNW@2018 / ResNetAMR / ICAMCNet.

**Launched 2026-07-14 ~09:53 UTC+8:** GPU0-only FastMLDNN paper siege
(`retune_model_siege.py --manifest siege_fastmldnn_10a_paper.json --gpu 0
--max-parallel 1 --paper-exact --promote`); log
`work_dirs/amr_benchmark_retune/siege_fastmldnn_paper.log`. ResNetAMR TF-port
siege on GPU0 stopped. JDM left on GPU1. Keepalive patched to avoid dual-GPU
siege while JDM owns GPU1.

Details: [`tf_alignment_audit.md`](./tf_alignment_audit.md) §0.0,
[`accuracy_tracking.md` Notes](./accuracy_tracking.md#notes-manual).

---

## Architecture freeze policy

**Core principle (non-negotiable):** retunes must **not** change model network
architecture. Layer topology, channel counts, backbone depth/width, conv kernel
sizes, LSTM/GRU hidden dimensions, head structure, and `num_classes` must stay
identical to the AMR-Benchmark Keras reference (or the paper where the reference
is silent). Hyperparameters, per-layer initialization, training strategy, and
documented input-pipeline divergences **are** fair game.

See also [`audit_changes.md`](./audit_changes.md) — past fixes (Xavier init,
per-sample L2 `SelfNormalize`, MCLDNN LSTM reshape, CNN4 kernel `(2,8)`,
DAE head wiring) were **bugfixes to match the Keras reference**, not architecture
changes. Do not use retune configs to re-open those decisions (e.g. removing
CLDNNW ZeroPad, shrinking conv stacks, or swapping backbone types).

| Category | Allowed in retune | Forbidden |
|----------|-------------------|-----------|
| **Init** | `model.backbone.init_cfg` (Xavier, Kaiming, LSTM/RNN orthogonal) | Adding/removing layers; changing channel dims |
| **Optim / schedule** | `optim_wrapper` (lr, wd, grad clip); `param_scheduler`; `train_cfg.max_epochs` | Changing optimizer *type* only if paper/AMC reference already uses it |
| **Early stopping** | `custom_hooks` (`EarlyStoppingHook` patience, `min_delta`; `[]` to disable) | — |
| **Batch / data** | `train_dataloader.batch_size`; swap dataset base for documented input modality (I/Q vs A/P) | Changing `frame_length` to alter LSTM input size beyond reference |
| **Input norm** | `SelfNormalize` / L2-norm pipeline bases (`iq-l2norm-*`) | Removing ZeroPad to “simplify” CLDNNW (already a fixed divergence) |
| **Loss weights** | Head `loss_weight` when head topology unchanged | Replacing `ClsHead` with a different head type |
| **Inference** | — (AMR is train-only) | Any backbone/head structural edit |

`tools/amr_benchmark/retune_sweep.py` documents permitted `--cfg-options` keys;
manifest configs under `configs/amr_benchmark/retune/` must respect this table.

---

## Pass criteria (tiered)

See [`goal_mode.md`](./goal_mode.md).

| Context | Overall | Peak |
|---------|---------|------|
| **Tracking** (all models) | measured ≥ target − **1.5 pp** | measured ≥ target − **1.0 pp** |
| **Campaign Tier A** (MLDNN / FastMLDNN / HCGDNN) | measured ≥ **paper target** | measured ≥ **paper target** |
| **Campaign Tier B** (TF ports) | measured ≥ target − **1.5 pp** | measured ≥ target − **1.0 pp** |
| **Split-adjusted triage** | Tier B RML only (secondary); **not** for own methods | — |

Best SNR is informational only. Default split **50/10/40**. Own methods: split is
paper-native.

---

## Triage of 39 fails (2026-07-05 baseline)

### Critical — dead training or >10 pp below target

Overall or peak near random chance, or overall gap ≥ 10 pp. Fix init / LR /
long-sequence stabilisation first.

| Model | Dataset | Overall (tgt → meas) | Peak (tgt → meas) | Gap | Likely cause |
|-------|---------|----------------------|-------------------|-----|--------------|
| **CNN1DPF** | RML2018.01A | 57 → **4.17** | 91 → **4.17** | −52.8 / −86.8 | Random-chance lock (`ln(24)`); missing Xavier; long-seq LR |
| **CGDNet** | RML2018.01A | 57 → **35.87** | 92 → **51.67** | −21.1 / −40.3 | Long-seq + default init; needs L2-norm base (already on) + LR warmup |
| **CLDNNW** | RML2018.01A | 55 → **37.19** | 88 → **53.33** | −17.8 / −34.7 | CLDNNW lacks auto-Xavier (unlike CLDNNL); long-seq |
| **FastMLDNN** | RML2016.10A | 63.24 → **61.02** (was stale 39.32) | 92 → **91.52** | −2.2 / −0.5 | Synced `esoff300`; paper-exact+tracking fail; **Tier A** — paper MultiStep+β next (not split blame) |
| **CNN2** | RML2018.01A | 58 → **42.35** | 92 → **65.23** | −15.7 / −26.8 | Long-seq CNN; raw IQ scale |
| **MCNet** | HisarMod | 70 → **56.03** | 97 → **79.00** | −14.0 / −18.0 | HisarMod + split divergence |
| **CLDNNL** | RML2018.01A | 57 → **46.67** | 92 → **81.40** | −10.3 / −10.6 | Long-seq; Xavier added but LR may still diverge |
| **ResNetAMR** | HisarMod | 80 → **72.49** | 100 → 98.00 | −7.5 / peak OK | HisarMod cluster |
| **PETCGDNN** | HisarMod | 75 → **64.68** | 99 → **85.32** | −10.3 / −13.7 | HisarMod RNN cluster |
| **DAE** | HisarMod | 40 → 54.27 | 70 → **61.39** | peak −8.6 | HisarMod + DAE head tuning |

### Marginal — within ~1–3 pp of pass threshold

Retune ES patience, LR ±2×, or input modality A/B.

| Model | Dataset | Failing metric | Gap (pp) | Proposed lever |
|-------|---------|----------------|----------|----------------|
| HCGDNN | RML2016.10A | overall | −0.86 | AMC lr=1e-3; ES patience 25–30 |
| LSTM2 | RML2016.10A | peak | −0.89 | ES patience 30; lr=5e-4; I/Q variant |
| ICAMCNet | HisarMod | peak | −0.44 | ES patience; already pass on overall |
| PETCGDNN | RML2018.01A | peak | −0.82 | L2-norm already on; ES patience |
| CLDNNL | RML2016.10A | peak | −0.86 | ES patience |
| CGDNet | RML2016.10A | overall | −0.95 | Already L2-norm; lr tweak |
| ResNetAMR | RML2016.10B | overall | −0.13 | ES patience (trivial) |
| ResNetAMR | RML2018.01A | overall | −0.26 | ES patience |
| MCNet | RML2016.10A | overall | −0.52 | ES patience |
| CNN1DPF | RML2016.10A | overall | −0.53 | ES patience |
| DensCNN | HisarMod | overall | −0.59 | HisarMod wave-2 |
| DensCNN | RML2016.10B | overall | −0.78 | ES patience |
| MCNet | RML2016.10B | overall | −1.01 | ES patience |
| DensCNN | RML2016.10A | overall | −1.02 | ES patience |
| ResNetAMR | RML2016.10A | overall | −1.17 | ES patience |
| GRU2 | RML2018.01A | peak | −1.28 | ES patience |
| CNN1DPF | RML2016.10B | overall | −2.08 | ES patience |
| MCLDNN | HisarMod | overall | −2.84 | HisarMod wave-2 |
| CNN4 | RML2016.10B | overall | −1.43 | ES patience |

### Structural — HisarMod cluster & RML2018.01A long-sequence band

Systematic underperformance on **HisarMod** (11 fails) and **RML2018.01A**
(10 fails). Root causes documented in audit:

1. **50/10/40 split vs paper 8:2:5 (Hisar)** — expect ~2–4 pp lower overall.
2. **1024-sample frames** — CLDNN/CGDNet/FastMLDNN need `frame_length=1024`,
   LR warmup, grad clip (see FastMLDNN 2018 fix in audit).
3. **HisarMod IQ length fix** applied; remaining gap is optimisation / ES.
4. **Input modality divergences** — LSTM2/CNN1DPF use A/P vs Keras I/Q.

**HisarMod fail cluster (11):** MCNet, ICAMCNet, ResNetAMR, DensCNN, GRU2,
LSTM2, DAE, MCLDNN, CLDNNW, CLDNNL, PETCGDNN.

**RML2018.01A fail band (10):** CNN2, CNN4, DensCNN, GRU2, ResNetAMR, CLDNNW,
CLDNNL, CGDNet, PETCGDNN, CNN1DPF.

---

## Prioritized retune queue

| Priority | Model × Dataset | Baseline gap | Wave | Interventions |
|----------|-----------------|--------------|------|---------------|
| **P1** | CNN1DPF × RML2018.01A | −52.8 pp overall | **Wave 1** | Xavier; lr 2e-4 + warmup + clip; Kaiming |
| **P2** | FastMLDNN × RML2016.10A | −23.9 pp overall | **Wave 1** | AMC lr 1e-3; lr 2e-4 + warmup; ES off 150 ep |
| **P3** | CLDNNW × RML2018.01A | −17.8 pp overall | **Wave 1** | Xavier; lr 2e-4 + warmup; ES patience 30 |
| **P4** | CGDNet × RML2018.01A | −21.1 pp overall | **Wave 1** | lr 2e-4 + warmup + clip; lr 5e-4; ES patience |
| **P5** | HCGDNN × RML2016.10A | −0.86 pp overall | **Wave 1** | AMC lr; ES patience 25–30 |
| **P5** | LSTM2 × RML2016.10A | −0.89 pp peak | **Wave 1** | lr 5e-4; ES patience; I/Q + L2 norm |
| P6 | CNN2 × RML2018.01A | −15.7 pp | Wave 2 | raw IQ ok; ES patience; optional L2 A/B |
| P7 | HisarMod cluster (11) | −3 to −14 pp | Wave 2–3 | per-model ES + lr; split-aware targets |
| P8 | Remaining 10A/10B marginal | −0.1 to −2 pp | Wave 3 | ES patience sweep only |

---

## Intervention catalog

Interventions are grouped by the [architecture freeze policy](#architecture-freeze-policy).

### Allowed

| Category | Options | When to apply | Config pattern |
|----------|---------|---------------|----------------|
| **Init** | Xavier uniform (Keras glorot) | Random-chance lock, deep conv/RNN | `model.backbone.init_cfg` in retune `.py` |
| | Kaiming fan-in | ReLU-heavy CNN (CNN1DPF) | `dict(type='Kaiming', layer='Conv1d', …)` |
| | Orthogonal / LSTM-RNN init | CGDNet GRU, LSTM2 | `dict(type='LSTM'/'RNN', layer=…, gain=1)` |
| **LR / schedule** | Adam 1e-3 (AMC default) | Baseline CNN sweep | inherit `_base_/schedules/amc.py` |
| | Adam 2e-4 + LinearLR warmup 5 ep | Long-seq divergence (2018) | see `fastmldnn_iq-ap-deepsig-201801A.py` |
| | Adam 5e-4 | Moderate halving | retune override |
| | CosineAnnealingLR T_max=150 | All models (fixed 2026-06-29) | already in AMC base |
| **L2 SelfNormalize** | On IQ pipeline | MCLDNN, CGDNet, CLDNNW, GRU2, PETCGDNN | `iq-l2norm-*` base configs |
| | Off (keep raw) | CNN2/4, CLDNNL, ResNetAMR | audit A/B table |
| **Batch size** | 400 vs 640 | FastMLDNN uses 640 | `train_dataloader.batch_size` |
| **Gradient clip** | max_norm=5.0 | Long-seq sum-merge (FastMLDNN, 2018) | `optim_wrapper.clip_grad` |
| **Early stopping** | min_delta 0.1, patience 15 | Default AMC | `_base_/runtimes/amc.py` |
| | min_delta 0.05, patience 25–30 | Marginal fails / slow 2018 convergence | retune override |
| | ES disabled | Suspected premature stop | `custom_hooks = []` |
| **Warmup** | LinearLR 5-epoch ramp | 2018 dead-ReLU / long-seq | `param_scheduler` list with `LinearLR` + cosine |
| **Input modality** | I/Q vs A/P | LSTM2, CNN1DPF documented divergence | swap dataset base or pipeline |
| **Weight decay** | Adam default (none) | Low priority | `optim_wrapper.optimizer.weight_decay` |
| **Loss weights** | Head `loss_weight` | DAE CE/MSE ratio (fixed in baseline) | `model.head.loss` |

### Forbidden

| Change | Why blocked | Where documented |
|--------|-------------|------------------|
| Backbone depth/width, conv kernel sizes | Alters AMR-Benchmark topology | `audit_changes.md` per-model table |
| LSTM/GRU hidden dims, extra/removed layers | Same | Keras reference parity |
| `num_classes` mismatch | Dataset-specific head width is structural | smoke tests in audit |
| Removing CLDNNW ZeroPad | Intentional CSRR divergence — do not retune away | `audit_changes.md` § Known divergences |
| Replacing backbone/head `type` | Architecture swap | — |
| MCLDNN reshape revert | Bugfix to match Keras LSTM time axis | `audit_changes.md` Phase 2 MCLDNN |

### `--cfg-options` quick overrides

For one-off probes without a new config file:

```bash
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/train.py configs/cldnnw/cldnnw_iq-deepsig-201801A.py \
  --work-dir work_dirs/amr_benchmark_retune/cldnnw/deepsig201801A/probe \
  --cfg-options optim_wrapper.optimizer.lr=2e-4
```

Nested lists (e.g. `param_scheduler`) are easier as dedicated files under
`configs/amr_benchmark/retune/`.

---

## Per-model siege mode

**Orchestrator:** `tools/amr_benchmark/retune_model_siege.py`  
**Queue:** `configs/amr_benchmark/retune/siege_queue.json`  
**Status artifact:** `work_dirs/amr_benchmark_retune/SIEGE_STATUS.json`

Siege mode resolves **one (model × dataset) at a time** — finish or exhaust the
current algorithm before starting the next. Within each siege, **all variants
screen in parallel** (up to `--max-parallel` on the GPU pool).

### Queue order (2026-07-14 reset — Tier A first)

| Priority | Pair | Gap vs paper | Rationale |
|----------|------|--------------|-----------|
| **P0** | FastMLDNN × RML2016.10A | −2.22 pp (best **61.02** / **91.52**) | **Paper-recipe siege** (`siege_fastmldnn_10a_paper.json`); paper_exact |
| **P1** | HCGDNN × RML2016.10A | −1.86 pp overall (peak OK) | Paper MultiStep / long budget; prior lr1e3 **hurt** |
| **P2** | MLDNN × RML2016.10A | **pass** paper-exact | Hold — no siege |
| **P3+** | LSTM2 / ICAMCNet / CLDNNW@2018 / … | Tier B | **approximate only**; deprioritized |

Stop futile ResNetAMR / ICAMCNet / CLDNNW paper-exact sieges on GPU0 while Tier A open.

### Overnight stall + recovery (2026-07-14 → 15)

| Issue | Fix |
|-------|-----|
| JDM `paper_exact` waiter PID **2919652** hung ~15.5h — `pgrep -f det_paper_exact_…` matched its own cmdline | Killed; `launch_paper_exact_keepalive.sh` now uses `train_running()` (requires `tools/train.py`) |
| `jdm_amc_launched=true` with no JDM process → GPU1 idle / AMR backfill blocked | `gpu_keepalive.sh` auto-clears stale flag via `jdm_gpu1_live()`; also treats `test_det.py` as ownership |
| GPU0 stuck on Tier-B ResNetAMR@2018 while Tier A open | Preempted; launched **HCGDNN paper MultiStep** (`siege_hcgdnn_10a_paper.json`) |
| FastMLDNN@10A paper siege (3 variants) finished 2026-07-14, still below 63.24 | Marked exhausted; HCGDNN is next Tier A priority |

**Running (2026-07-15 ~10:33):** HCGDNN `paper_multistep_esoff800` GPU0; JDM merge 5-ep det + wave3b AMC AWGN joint GPU1 (`joint_awgn_5ep_amc.log`).

### Per-pair workflow

```
for entry in siege_queue (priority order):
    skip if status in {passed, exhausted, skipped}
    mark entry running
    launch ALL variants in entry.manifest IN PARALLEL (--max-parallel N)
    if any variant meets campaign_success (paper-exact, e.g. FastMLDNN ≥ 63.24% / 92.0%):
        mark entry passed; optionally --promote
        if --until-pass: stop pair (cancel pending variants)
    else:
        mark entry exhausted
    advance to next entry
```

### Contrast with goal-mode serial sweep

| Aspect | Goal mode (`retune_sweep.py`) | Siege mode (`retune_model_siege.py`) |
|--------|-------------------------------|--------------------------------------|
| Pair order | Manifest priority, all pairs in one run | Explicit `siege_queue.json`; one pair resolved before next |
| Variants within pair | **Serial** — train variant 1, then 2, … | **Parallel** — up to N variants on GPU pool |
| Stop pair | `--until-pass` after first pass | Same (`--until-pass`, default on) |
| Manifest | Single wave manifest | Per-pair siege manifest + queue file |

### Launch

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
mkdir -p work_dirs/amr_benchmark_retune

# Dry-run
python tools/amr_benchmark/retune_model_siege.py \
  --queue configs/amr_benchmark/retune/siege_queue.json --dry-run

# Full siege (2 GPUs, parallel screening)
nohup python tools/amr_benchmark/retune_model_siege.py \
  --queue configs/amr_benchmark/retune/siege_queue.json \
  --gpu 0,1 --max-parallel 2 --until-pass --promote \
  >> work_dirs/amr_benchmark_retune/siege.log 2>&1 &
echo "siege PID=$!"
```

Single-pair probe (FastMLDNN 10A only):

```bash
python tools/amr_benchmark/retune_model_siege.py \
  --manifest configs/amr_benchmark/retune/siege_fastmldnn_10a.json \
  --gpu 0,1 --max-parallel 2 --until-pass
```

**Coexistence with Wave 1 resume:** if `wave1_resume` orchestrator (PID 1211790)
is still running, do **not** launch siege on the same GPUs — wait for completion
or stop the resume sweep first. As of 2026-07-09 11:01 UTC+8, PID 1211790 is
**not running**; GPUs idle → siege may launch immediately.

---

## Goal mode usage

Goal mode keeps tuning until targets are met or the queue is exhausted.
See [`goal_mode.md`](./goal_mode.md) for the full spec.

```bash
# Status only (no GPU) — parse tracking table + goals.json
python tools/amr_benchmark/retune_sweep.py --goal-status \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json

# Run until all fails resolved or queue exhausted
python tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json \
  --goal-mode --stop-when-all-pass --gpu 0,1 --max-parallel 2

# Single pair: try variants until pass
python tools/amr_benchmark/retune_sweep.py \
  --model cnn1dpf --dataset deepsig201801A \
  --variants xavier_lr1e3,lr2e4_warmup_clip,kaiming_selu \
  --goal-mode --until-pass --gpu 0
```

Status artifact: `work_dirs/amr_benchmark_retune/GOAL_STATUS.json`

---

## Running retunes

### Dry-run (recommended first)

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json \
  --dry-run
```

### Wave 1 — full manifest (2 GPUs, 2 parallel 10A jobs + 1 long 2018 job)

```bash
# Terminal A — fast 10A retunes (both GPUs)
CUDA_VISIBLE_DEVICES=0,1 \
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json \
  --gpu 0,1 --max-parallel 2 \
  2>&1 | tee work_dirs/amr_benchmark_retune/wave1.log

# Or single pair:
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --model fastmldnn --dataset deepsig201610A \
  --variants lr1e3,lr2e4_warmup,es_off150 --gpu 0
```

### Promote a passing variant to baseline tracking

```bash
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --model hcgdnn --dataset deepsig201610A --variants lr1e3 \
  --gpu 0 --promote --force
```

Copies `best_*.pth` and `res/paper.pkl` into
`work_dirs/amr_benchmark/<model>/<dataset>/` and refreshes the auto table in
`accuracy_tracking.md`.

---

## Success criteria & tracking updates

1. **Tracking pass** (auto table only): overall ≥ target − 1.5 pp **and** peak ≥
   target − 1.0 pp — logic unchanged in `run_migration.py::_classify`.
2. **Campaign success** (goal/siege stop): overall ≥ **paper target** **and** peak ≥
   **paper target** (no tolerance); see `configs/amr_benchmark/retune/goals.json`.
   FastMLDNN @ RML2016.10A: **63.24%** overall, **92.0%** peak.
3. **Log:** every sweep appends rows to [`retune_results.md`](./retune_results.md).
4. **Promote:** use `--promote` on the best variant; baseline table updates via
   `run_migration.py` renderer (tracking pass/fail logic).
5. **Campaign doc:** update the **Wave results** section below after each wave.
6. **Do not** hand-edit rows between `AMR_BENCHMARK_AUTO_TABLE_BEGIN/END`.

---

## Wave 1 — first-wave execution (2026-07-06)

**GPUs:** 2× RTX 3090 (idle at launch).  
**Manifest:** `configs/amr_benchmark/retune/wave1_manifest.json` (17 experiments).  
**Work root:** `work_dirs/amr_benchmark_retune/`

| Experiment | Variant config | Status | Overall | Peak | Notes |
|------------|----------------|--------|---------|------|-------|
| CNN1DPF @ 2018 | xavier_lr1e3 | _running_ | — | — | P1 (GPU0, resumed 2026-07-08) |
| CNN1DPF @ 2018 | lr2e4_warmup_clip | **pass** | 55.95 | 90.87 | P1 — **promoted 2026-07-08** |
| CNN1DPF @ 2018 | kaiming_selu | fail | 55.45 | — | P1 (peak-only miss) |
| FastMLDNN @ 10A | lr1e3 | _queued_ | — | — | P2 |
| FastMLDNN @ 10A | lr2e4_warmup | _queued_ | — | — | P2 |
| FastMLDNN @ 10A | es_off150 | _queued_ | — | — | P2 |
| CLDNNW @ 2018 | xavier_lr2e4 | fail | 43.80 | 65.65 | P3 — all 3 variants exhausted |
| CLDNNW @ 2018 | xavier_lr1e3 | fail | 41.36 | 61.16 | P3 |
| CLDNNW @ 2018 | es_patience30 | fail | 42.04 | 61.96 | P3 |
| CGDNet @ 2018 | lr2e4_warmup | fail | 46.83 | 71.45 | P4 — all 3 variants exhausted |
| CGDNet @ 2018 | lr5e4 | fail | 49.58 | 75.86 | P4 (best overall) |
| CGDNet @ 2018 | es_patience30 | fail | 45.90 | 69.72 | P4 |
| HCGDNN @ 10A | lr1e3 | _queued_ | — | — | P5 marginal |
| HCGDNN @ 10A | es_patience25 | _queued_ | — | — | P5 |
| LSTM2 @ 10A | lr5e4 | _queued_ | — | — | P5 peak-only |
| LSTM2 @ 10A | es_patience30 | _queued_ | — | — | P5 |
| LSTM2 @ 10A | iq_input | _queued_ | — | — | P5 I/Q path |

**Passes converted from fail:** **1** — first fail→pass achieved **2026-07-08**:
`cnn1dpf/deepsig201801A` via `lr2e4_warmup_clip` (overall 55.95%, peak 90.87%).
Tracking synced: **23 pass / 38 fail** (was 22/39). See [`retune_results.md`](./retune_results.md).

### Wave 1 resumed (2026-07-06 15:59)

JDM P0 30-ep detectors finished; GPUs confirmed free. Wave 1 relaunched in
goal mode:

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
mkdir -p work_dirs/amr_benchmark_retune
nohup /home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json \
  --gpu 0,1 --max-parallel 2 \
  --goal-mode --stop-when-all-pass \
  >> work_dirs/amr_benchmark_retune/wave1.log 2>&1 &
echo "retune_sweep PID=$!"
```

**Resume PID:** `241810` (orchestrator, goal-mode + stop-when-all-pass)  
**First job after resume:** `cgdnet/deepsig201801A/es_patience30` (train PID 241859, GPU 0)  
**Log:** `work_dirs/amr_benchmark_retune/wave1.log`

Monitor:

```bash
tail -f work_dirs/amr_benchmark_retune/wave1.log
grep -E 'PASS|FAIL|done' work_dirs/amr_benchmark_retune/wave1.log
ps -p 241810 -o pid,cmd
```

---

## FastMLDNN Wave 2 plan (2026-07-08)

**Gap analysis:** [FastMLDNN gap analysis](fc5c869c-00b2-47f8-9ff0-b0dedeab91cc) / [`fastmldnn_paper_comparison.md`](./fastmldnn_paper_comparison.md)  
**Paper target (RML2016.10A):** overall **63.24%**, peak **≈92% @ 16 dB**  
**Baseline sweep:** 39.32% overall / 62.61% peak (−23.9 / −29.4 pp)  
**Wave-1 best (`es_off150`):** 51.89% overall (+12.6 pp vs baseline, still −11.4 pp vs paper)

Architecture is frozen; Wave 2 applies only hyperparameter / init / pipeline / training-strategy fixes identified in the paper comparison (§4.2, §6):

| Priority | Fix | Wave-2 variant | Rationale |
|----------|-----|----------------|-----------|
| **P0** | `head.beta=0.5` | all 4 | Restore paper multi-loss (center distance expansion); baseline uses `beta=0` |
| **P0** | Xavier + TruncNormal init | all 4 | Paper `init_cfg` on Conv1d / Linear |
| **P0** | ES disabled (`custom_hooks=[]`) | all 4 | Wave-1 `es_off150` +12.6 pp; strict ES stops ~epoch 41 |
| **P1** | IQ `SelfNormalize` L2 | `beta05_xavier_l2_esoff`, combo | MCLDNN sibling passes at 62% with L2; FastMLDNN shares tiny IQ scale |
| **P1** | `backbone.dp=0.07` | `beta05_xavier_dp007_esoff`, combo | Paper channel-mode pretrain value (default AMR `dp=0.5`) |
| **P2** | Hisar / 201801A stabilisation | *Wave 3* | lr 2e-4 + warmup + grad clip (201801A fix template); defer until 10A converges |

**Manifest:** `configs/amr_benchmark/retune/wave2_fastmldnn_manifest.json` (4 experiments, **deepsig201610A only**).

| ID | Variant | Interventions beyond P0 base |
|----|---------|------------------------------|
| `W2_fastmldnn_10a_beta05_xavier_esoff150` | `beta05_xavier_esoff150` | P0 only |
| `W2_fastmldnn_10a_beta05_xavier_l2_esoff` | `beta05_xavier_l2_esoff` | + IQ L2 |
| `W2_fastmldnn_10a_beta05_xavier_dp007_esoff` | `beta05_xavier_dp007_esoff` | + dp=0.07 |
| `W2_fastmldnn_10a_beta05_xavier_l2_dp007_esoff` | `beta05_xavier_l2_dp007_esoff` | + L2 + dp=0.07 (full stack) |

**Campaign success criterion:** overall ≥ **63.24%** **and** peak ≥ **92.0%**
(paper-reported; no tolerance). Tracking pass would be ≥ 61.74% / 91.0%.

### Wave 2 — launch (do not run while Wave 1 sweep active)

**GPU status (2026-07-08 16:25):** GPU 0 — Wave 1 `lstm2/deepsig201610A/lr5e4` (train PID 1055856). GPU 1 idle before launch. Wave 1 orchestrator PID **241810** still running (unchanged).

**Wave 2 launched (2026-07-08 16:25 UTC+8):** `retune_sweep.py` orchestrator PID **1058806** — `--gpu 1 --max-parallel 1 --goal-mode --until-pass`; log `work_dirs/amr_benchmark_retune/wave2_fastmldnn.log`. First job: `beta05_xavier_esoff150` on GPU 1.

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
mkdir -p work_dirs/amr_benchmark_retune

# Dry-run first
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave2_fastmldnn_manifest.json \
  --dry-run

# Full Wave 2 (both GPUs, 2 parallel 10A jobs)
nohup /home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave2_fastmldnn_manifest.json \
  --gpu 0,1 --max-parallel 2 \
  --goal-mode --until-pass \
  >> work_dirs/amr_benchmark_retune/wave2_fastmldnn.log 2>&1 &
echo "wave2_fastmldnn PID=$!"
```

Or single-pair probe:

```bash
/home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --model fastmldnn --dataset deepsig201610A \
  --variants beta05_xavier_esoff150,beta05_xavier_l2_dp007_esoff \
  --gpu 0 --goal-mode --until-pass
```

**Note:** `--variants` CLI resolves `wave1_<model>_<dataset>_<variant>.py` only; for Wave 2 use `--manifest` (configs are prefixed `wave2_`).

### Wave 2 — results (2026-07-08, completed 18:43 UTC+8)

**Orchestrator PID:** 1058806 (log `work_dirs/amr_benchmark_retune/wave2_fastmldnn.log`).  
**Outcome:** all 4 variants **fail** — pair exhausted; best variant below.

| Variant | Overall | Peak | Status | Notes |
|---------|---------|------|--------|-------|
| `beta05_xavier_esoff150` | 53.69 | 82.84 | fail | P0 only (+14.4 / +20.2 pp vs baseline) |
| `beta05_xavier_l2_dp007_esoff` | **60.12** | **89.48** | fail | **Best Wave-2** — −3.12 / −2.52 pp vs **paper** 63.24 / 92.0 |
| `beta05_xavier_l2_esoff` | 55.55 | 83.91 | fail | L2 without dp=0.07 |
| `beta05_xavier_dp007_esoff` | 58.46 | 88.07 | fail | dp=0.07 without L2 |

**Wave-1 best for comparison:** `es_off150` → 51.89% overall / 82.91% peak.  
**Wave-2 lift (best combo):** +8.2 pp overall, +6.6 pp peak vs Wave-1 best.  
**Campaign success criterion:** overall ≥ **63.24%** **and** peak ≥ **92.0%** —
neither met; Wave-2 best still −3.12 pp below paper overall.

Details in [`retune_results.md`](./retune_results.md) (Run 2026-07-08 08:59:50).

### FastMLDNN Wave 3 — micro-tune plan (config only, not launched)

Best Wave-2 config (`beta05_xavier_l2_dp007_esoff`) is **2.54 pp** short of paper
overall (Wave-3 best **60.70%** on `esoff200` vs paper **63.24%**).
Wave 3 extends the winning stack with training-duration / schedule levers only (architecture frozen):

| Priority | Variant (proposed) | Interventions | Rationale |
|----------|-------------------|---------------|-----------|
| **W3-1** | `beta05_xavier_l2_dp007_esoff200` | Wave-2 stack + `max_epochs=200`, ES off | Val acc still climbing at epoch 115–141; paper uses 400 ep |
| **W3-2** | `beta05_xavier_l2_dp007_esoff_stepLR` | Wave-2 stack + StepLR (×0.1 @ ep 100, 150) | Paper Keras `lr_config` drops; cosine-only may under-shoot peak SNR |
| **W3-3** | `beta05_xavier_l2_dp007_esoff_lr5e4` | Wave-2 stack + lr=5e-4 (half paper 4.4e-4) | Marginal peak lift without destabilising overall |

**Manifest:** `configs/amr_benchmark/retune/siege_fastmldnn_10a.json` (Wave-2 + Wave-3 variants).  
**Configs created:** `wave3_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff{200,stepLR,lr5e4}.py`  
**Launch via siege:** see [Per-model siege mode](#per-model-siege-mode).

### FastMLDNN siege round 2 (launched 2026-07-10)

**Manifest:** `configs/amr_benchmark/retune/siege_fastmldnn_10a_r2.json` (4 variants: fixed `stepLR`, reuse `esoff200`, `esoff250`, `lr3e4`).  
**Campaign success target:** overall ≥ **63.24%**, peak ≥ **92.0%** (paper-exact).
Wave-3 best so far: **60.70** / **90.64** on `esoff200` (−2.54 / −1.36 pp vs paper).
**Orchestrator PID:** `1396373` — log `work_dirs/amr_benchmark_retune/siege_r2.log` (`--gpu 0,1 --max-parallel 2 --until-pass --promote`).

```bash
nohup /home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_model_siege.py \
  --manifest configs/amr_benchmark/retune/siege_fastmldnn_10a_r2.json \
  --gpu 0,1 --max-parallel 2 --until-pass --promote \
  >> work_dirs/amr_benchmark_retune/siege_r2.log 2>&1 &
```

### FastMLDNN siege round 2 — results (2026-07-10, completed)

**Orchestrator PID:** 1396373 — log `work_dirs/amr_benchmark_retune/siege_r2.log`.  
**Outcome:** all 4 variants **fail** — pair still exhausted; paper target **not met**.

| Variant | Overall | Peak | vs paper (63.24 / 92.0) | Status | Notes |
|---------|---------|------|-------------------------|--------|-------|
| `beta05_xavier_l2_dp007_esoff250` | **60.90** | **91.18** | −2.34 / −0.82 pp | fail | **Best siege r2** — best val @ ep 232 |
| `beta05_xavier_l2_dp007_esoff200` | 60.70 | 90.64 | −2.54 / −1.36 pp | fail | Reused ckpt (siege r1 best) |
| `beta05_xavier_l2_dp007_esoff_lr3e4` | 59.14 | 88.00 | −4.10 / −4.00 pp | fail | lr=3e-4 underperforms esoff250 |
| `beta05_xavier_l2_dp007_esoff_stepLR` | — | — | — | **error** | `MultiStepParamScheduler` got unexpected `T_max` — inherited CosineAnnealingLR keys merged into MultiStepLR dict |

**Siege r1 best for comparison:** `esoff200` → 60.70% / 90.64%.  
**Siege r2 lift (best):** +0.20 pp overall, +0.54 pp peak vs r1 best.  
**Campaign success criterion:** overall ≥ **63.24%** **and** peak ≥ **92.0%** — neither met; best still −2.34 pp below paper overall.

**stepLR root cause:** MMEngine config merge retains base `T_max`/`eta_min` from `CosineAnnealingLR` when only `type`/`milestones`/`gamma` are overridden. Fix: `_delete_=True` on `param_scheduler` dict (see fixed config).

### FastMLDNN siege round 3 (launched 2026-07-11 13:53 UTC+8)

Best siege r2 (`esoff250`) is **2.34 pp** short of paper overall, **0.82 pp** short on peak.
Round 3 extends the winning stack (beta=0.5 + Xavier + L2 + dp=0.07 + ES off) with
training-duration / schedule levers only (architecture frozen):

| Priority | Variant | Interventions | Rationale |
|----------|---------|---------------|-----------|
| **R3-1** | `beta05_xavier_l2_dp007_esoff_stepLR` | Fixed MultiStepLR (`_delete_=True`, ×0.1 @ ep 100, 150) | Paper Keras step decay; prior run errored on merged `T_max` |
| **R3-2** | `beta05_xavier_l2_dp007_esoff300` | Wave-2 stack + `max_epochs=300`, cosine T_max=300 | esoff250 still climbing @ ep 232; paper uses 400 ep |
| **R3-3** | `beta05_xavier_l2_dp007_esoff250_ft50_lr1e4` | `load_from` esoff250 best + 50 ep @ lr=1e-4 | Marginal polish from best checkpoint without full retrain |

**Manifest:** `configs/amr_benchmark/retune/siege_fastmldnn_10a_r3.json` (3 variants).  
**Configs created:** `wave3_fastmldnn_deepsig201610A_beta05_xavier_l2_dp007_esoff{300,250_ft50_lr1e4}.py` + fixed `…_esoff_stepLR.py`.  
**Pre-launch checks (2026-07-11):** `py_compile` on all 3 r3 configs; stepLR `param_scheduler` verified (no merged `T_max`). GPUs idle ~27 h after siege r2 ended.

**Siege r3 launched (2026-07-11 13:53 UTC+8):** orchestrator PID **1501839** — `--gpu 0,1 --max-parallel 2 --until-pass --paper-exact --promote`; log `work_dirs/amr_benchmark_retune/siege_r3.log`. First parallel jobs: `esoff250_ft50_lr1e4` (GPU 0), `esoff300` (GPU 1); `stepLR` queued after a slot frees.

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
mkdir -p work_dirs/amr_benchmark_retune
nohup /home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_model_siege.py \
  --manifest configs/amr_benchmark/retune/siege_fastmldnn_10a_r3.json \
  --gpu 0,1 --max-parallel 2 --until-pass --paper-exact --promote \
  >> work_dirs/amr_benchmark_retune/siege_r3.log 2>&1 &
```

---

## Continuous GPU utilization

**Policy: zero idle GPUs.** A background scheduler keeps both RTX 3090s busy
without manual progress checks. The user monitors `scheduler.log` only.

**Proactive health inspection:** background `health_watchdog.sh` (3 min) plus
optional `run_inspection.sh` (cron or manual). See
[`operations.md` § 主动巡检](./operations.md#主动巡检-proactive-inspection)
for log paths, checks, and auto-remediation.

| Component | Path |
|-----------|------|
| **Scheduler daemon** | `tools/amr_benchmark/gpu_keepalive.sh` |
| **Scheduler log** | `work_dirs/amr_benchmark_retune/scheduler.log` |
| **Scheduler state** | `work_dirs/amr_benchmark_retune/scheduler_state.json` |
| **Primary siege queue** | `configs/amr_benchmark/retune/siege_queue.json` |
| **Full pipeline queue** | `configs/amr_benchmark/retune/siege_queue_full.json` |

### Scheduler behaviour (every 5 min)

1. Count AMR / JDM `train.py` jobs per GPU via `nvidia-smi` + `/proc` cmdline.
2. **Never kill** running jobs.
3. If **both GPUs free** and siege r3 variants incomplete → resume
   `siege_fastmldnn_10a_r3.json`.
4. If **no siege orchestrator** and a GPU slot is free → launch next **pending**
   entry from `siege_queue.json` (`--paper-exact --promote`). When only GPU0 is
   busy, launch the next pair on **GPU1 alone** (`--gpu 1 --max-parallel 1`).
5. When primary queue is exhausted → goal-mode `wave1_manifest.json` resume.
6. **JDM secondary slot:** when AMR holds GPU0 only and GPU1 AMR-idle **>10 min**,
   auto-start JDM wave3 **Track B** on GPU1 (`wave3_trackb_manifest.json`).

### GPU policy

| GPU | Role | Workload |
|-----|------|----------|
| **GPU 0** | AMR primary | Siege orchestrator, parallel variant screening |
| **GPU 1** | AMR parallel **or** JDM secondary | Second siege variant when 2-up; else JDM Track B after 10 min idle |

### Full pipeline order (`siege_queue_full.json`)

After FastMLDNN siege r3 completes:

| Priority | Pair | Gap | Variants |
|----------|------|-----|----------|
| P1 | HCGDNN × RML2016.10A | −0.86 pp overall | 2 (lr1e3, es25) |
| P2 | LSTM2 × RML2016.10A | −0.89 pp peak | 3 (lr5e4, es30, iq) |
| P3 | ICAMCNet × HisarMod | −1.44 pp peak | 1 (es30) |
| P4–P19 | Remaining marginals | −0.13 to −2.84 pp | goal-mode / future siege manifests |
| P20 | Critical 2018 + Hisar band | >3 pp | `wave1_manifest.json` goal-mode |

Marginal pairs are ordered by gap analysis (closest to paper target first).

### Launch scheduler (resilient wrapper)

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
mkdir -p work_dirs/amr_benchmark_retune
chmod +x tools/amr_benchmark/gpu_keepalive.sh
nohup bash tools/amr_benchmark/gpu_keepalive.sh \
  >> work_dirs/amr_benchmark_retune/scheduler.log 2>&1 &
echo "scheduler PID=$!"
```

Monitor (no manual GPU checks needed):

```bash
tail -f work_dirs/amr_benchmark_retune/scheduler.log
grep ACTION work_dirs/amr_benchmark_retune/scheduler.log
```

The scheduler runs an immediate tick on start, then every 300 s. It survives
workspace disconnects via `nohup` and appends all decisions to `scheduler.log`.

---

## Wave 1 resumed (2026-07-09 09:43 UTC+8)

GPUs idle after AMR/JDM sweeps ended. Wave 1 relaunched in goal mode (14 cached variants + 1 incomplete train):

```bash
cd /home/citybuster/Projects/ChangShuoRadioRecognition
mkdir -p work_dirs/amr_benchmark_retune
nohup /home/citybuster/Applications/conda/envs/ChangShuoRadioRecognition/bin/python \
  tools/amr_benchmark/retune_sweep.py \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json \
  --gpu 0,1 --max-parallel 2 \
  --goal-mode --stop-when-all-pass \
  >> work_dirs/amr_benchmark_retune/wave1_resume.log 2>&1 &
echo "retune_sweep PID=$!"
```

**Resume PID:** `1211790` (orchestrator, goal-mode + stop-when-all-pass)  
**Skip behaviour:** existing checkpoints + `res/paper.pkl` are reused (no retrain/retest); only missing runs train fresh.  
**First fresh job:** `fastmldnn/deepsig201610A/lr1e3` (prior Wave-1 run errored before train completed).  
**Log:** `work_dirs/amr_benchmark_retune/wave1_resume.log`

Monitor:

```bash
tail -f work_dirs/amr_benchmark_retune/wave1_resume.log
grep -E 'PASS|FAIL|exhausted|reusing' work_dirs/amr_benchmark_retune/wave1_resume.log
ps -p 1211790 -o pid,cmd
```

---

## Wave 2+ (planned)

- CNN2 / DensCNN / ResNetAMR @ RML2018.01A — ES patience + optional lr 5e-4
- HisarMod cluster — batch retune with split-adjusted expectations
- CLDNNL @ 2018 — lr warmup stack (mirror CLDNNW P3 recipe)
- Remaining marginal 10A/10B — automated ES patience grid (`min_delta` × `patience`)

**Wave 1 resumed 2026-07-06 15:59** — orchestrator PID `241810` (goal-mode); see [Wave 1 resumed](#wave-1-resumed-2026-07-06-1559).

**Paused jobs (2026-07-06 morning):** SIGTERM sent to `retune_sweep.py` (PID 188487) and in-flight Wave 1 trains — `wave1_cnn1dpf_deepsig201801A_kaiming_selu` (PID 188553), `wave1_cnn1dpf_deepsig201801A_lr2e4_warmup_clip` (PID 188554). Partial work dirs remain under `work_dirs/amr_benchmark_retune/cnn1dpf/deepsig201801A/`.

---

## ICAMCNet × HisarMod — peak-100 exhaustion (2026-07-14)

**Decision:** mark `siege_icamcnet_hisar` **exhausted** / `waived_peak_near_miss` (stop sieging).

| Item | Value |
|------|-------|
| Paper target | overall ~80 / **peak 100** |
| Last best (siege) | **82.27 / 98.56** (`es_patience30`; tracking peak also 98.56) |
| Observed ceiling | peak stuck ~98.45–98.56 across many `--force` ES/patience + `lr2e4_warmup` reruns |
| Gap to paper peak | ~1.44–1.55 pp; never closes under current recipe |

**Cause of GPU0 trap:** `health_watchdog.sh` stale-log recovery relaunched `siege_icamcnet_hisar.json` with `--force` after each failed train (~14 cycles). Peak never reaches paper 100; further identical ES/patience variants are **futile**.

**Action taken:** SIGTERM icamcnet orch+train on GPU0 only (JDM AMC on GPU1 untouched). Queue waiver recorded. Keepalive/watchdog now skip `--force` loops when a pair is exhausted/waived or the same paper-exact metrics already failed ≥3 times.

**Escalate (future):** beyond the same ES/patience / lr2e4_warmup family — architecture/recipe change, different head/optimizer, or paper-protocol audit — not another `--force` siege of `es_patience30`.
