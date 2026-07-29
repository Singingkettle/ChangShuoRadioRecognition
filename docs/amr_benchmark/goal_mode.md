# AMR-Benchmark Goal Mode

Goal mode keeps the retune campaign running until measurable targets are met
(or the priority queue is exhausted), instead of stopping after a fixed
manifest pass regardless of outcome.

**Orchestrators:**

| Mode | Tool | When to use |
|------|------|-------------|
| Goal mode (serial) | `tools/amr_benchmark/retune_sweep.py` | Walk a wave manifest; variants **serial** within each pair |
| **Siege mode (parallel)** | `tools/amr_benchmark/retune_model_siege.py` | Resolve **one pair at a time**; variants **parallel** for screening |

**Goals config:** `configs/amr_benchmark/retune/goals.json`  
**Status artifacts:** `work_dirs/amr_benchmark_retune/GOAL_STATUS.json`, `SIEGE_STATUS.json`  
**Campaign plan:** [`retune_campaign.md`](./retune_campaign.md)  
**Own-methods alignment:** [`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md)  
**Split / TF audit:** [`tf_alignment_audit.md`](./tf_alignment_audit.md)

---

## Two-tier campaign policy (2026-07-14)

| Tier | Models | `campaign_mode` | Success criterion |
|------|--------|-----------------|-------------------|
| **A — Own methods** | MLDNN, FastMLDNN, HCGDNN | `paper_exact` | measured ≥ **paper target** (no −1.5/−1.0) on every dataset with paper targets (esp. 10A) |
| **B — AMR-Benchmark ports** | all other models | `approximate` | measured ≥ target − **1.5 pp** overall / − **1.0 pp** peak (“差不多就行”) |

Configured in `goals.json` as `campaign_mode: "tiered"` with `tiers.A_own_methods` /
`tiers.B_amr_benchmark_ports` and `pair_campaign_modes` for 10A own-method pairs.

**Siege firepower:** stay on Tier A until paper-exact (or waiver). Deprioritize
CLDNNW@2018, ICAMCNet Hisar loops, and other TF ports to approximate-only work.

---

## Data split (frozen)

`goals.json` records `split_protocol: "50/10/40"` (train/val/test = **5:1:4**).

- **Tier A:** 5:1:4 is **paper-native** for MLDNN / FastMLDNN / HCGDNN. Do **not**
  blame residual gaps on “TF split concession.” Close gaps via paper architecture
  freeze + paper training recipe (schedule, multi-loss β, init, epochs, ES).
- **Tier B:** CSRR stays on 5:1:4; TF AMR-Benchmark RML used **6:2:2**. Small
  remaining gaps under approximate mode may cite that protocol difference.

Do not change the default split to TF 6:2:2 to make Tier B look closer.

---

## Architecture freeze policy

Goal mode does **not** relax the architecture freeze. Retunes may only change
hyperparameters, initialization, training strategy, and documented input-pipeline
choices. See [`retune_campaign.md` § Architecture freeze](./retune_campaign.md#architecture-freeze-policy).

---

## Goal semantics (do not conflate)

### 1. Tracking pass/fail (tolerance band)

Used by `accuracy_tracking.md` auto table and `run_migration.py::_classify`.
**Unchanged** by goal mode — still the authoritative status for the benchmark
matrix.

| Metric | Pass rule |
|--------|-----------|
| Overall test accuracy | measured ≥ target − **1.5 pp** |
| Peak accuracy (best SNR) | measured ≥ target − **1.0 pp** |
| Best SNR | informational only |

Example (FastMLDNN @ RML2016.10A): tracking **pass** at overall ≥ **61.74%**
and peak ≥ **91.0%**. Synced retune best **61.02** is still a tracking **fail**.

### 2. Campaign success — Tier A paper-exact

For MLDNN / FastMLDNN / HCGDNN (or any pair with
`pair_campaign_modes[…] = "paper_exact"`):

| Metric | Success rule |
|--------|--------------|
| Overall | measured ≥ **paper target** |
| Peak | measured ≥ **paper target** |

Example (FastMLDNN @ 10A): stop siege only at ≥ **63.24%** / ≥ **92.0%**.

### 3. Campaign success — Tier B approximate

For all other models when `campaign_mode: "tiered"`:

| Metric | Success rule |
|--------|--------------|
| Overall / peak | same as tracking (−1.5 / −1.0 pp) |

TF contrast is still useful for debugging; remaining small gaps after that band
may be accepted as approx OK (including 5:1:4 vs TF 6:2:2).

### 4. Split-adjusted triage (Tier B only)

`split_adjusted_targets` (paper − 2.5 pp) is **secondary triage for Tier B RML
ports only**. It does **not** apply to Tier A own methods and does **not**
replace paper-exact for FastMLDNN/HCGDNN/MLDNN.

---

## Campaign goal

1. **Primary:** bring Tier A own methods to **paper-exact** on 10A (and any
   other dataset with paper targets), documented in
   [`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md).
2. **Secondary:** convert Tier B tracking fails toward **approximate** pass, or
   document waiver / approx OK.

Baseline (2026-07-05): 39 fails, 21 passes, 20 measured-only rows.

---

## Stop conditions

| Mode | When tuning stops |
|------|-------------------|
| Default (no goal mode) | After scheduled manifest / variant list completes |
| `--goal-mode --until-pass` | Current pair meets **tier campaign success** **or** variants exhausted |
| `--goal-mode --stop-when-all-pass` | Tracking table shows **0 fails** |
| Queue exhausted, fails remain | Mark pair **exhausted**; escalate in campaign doc |

Siege mode uses the same per-pair campaign-success criterion
(`tools/goal_mode_helpers.py` → `resolve_pair_campaign_mode`).

---

## Auto-continue behaviour

### Goal mode (`retune_sweep.py`) — serial variants

```
for each (model, dataset) in manifest priority order:
    for variant in variants[(model, dataset)] sorted by priority:
        train → test → classify (tracking) + campaign check (tier-aware)
        if campaign_success:
            record goal_met=true; optionally --promote
            if --until-pass: break inner loop
        else:
            record goal_met=false; try next variant
    if all variants failed:
        mark pair exhausted → escalate
```

### Siege mode (`retune_model_siege.py`) — parallel screening

Resolve one (model × dataset) before advancing. Prefer Tier A manifests
(`siege_fastmldnn_10a_paper.json`, then HCGDNN paper-recipe) over Tier B ports.

**Queue file:** `configs/amr_benchmark/retune/siege_queue.json`  
**Paper FastMLDNN:** `siege_fastmldnn_10a_paper.json`

---

## CLI flags

### Goal mode (`retune_sweep.py`)

| Flag | Purpose |
|------|---------|
| `--goal-mode` | Enable goal-driven variant loop |
| `--until-pass` | Stop at first **campaign-success** variant (tier-aware) |
| `--stop-when-all-pass` | Stop when tracking shows 0 fails |
| `--goal-status` | Print status; no GPU |
| `--goals PATH` | Override goals JSON |
| `--paper-exact` | Force paper-exact for this run (overrides tier default) |
| `--no-paper-exact` | Force approximate / tolerance success |

### Siege mode (`retune_model_siege.py`)

Same `--paper-exact` / `--no-paper-exact` / `--goals` semantics; prefer launching
Tier A with `--paper-exact --promote` on GPU0 only when free (leave JDM on GPU1).

---

## Example commands

```bash
# Dry status check (no GPU)
python tools/amr_benchmark/retune_sweep.py --goal-status \
  --manifest configs/amr_benchmark/retune/wave1_manifest.json

# Tier A: FastMLDNN paper-recipe siege (paper-exact)
python tools/amr_benchmark/retune_model_siege.py \
  --manifest configs/amr_benchmark/retune/siege_fastmldnn_10a_paper.json \
  --gpu 0 --max-parallel 1 --until-pass --paper-exact --promote
```

---

## Artifacts

### `GOAL_STATUS.json`

Includes `campaign_mode` from goals (may be `"tiered"`). Per-pair success still
recorded via `goal_met` on each retune row.

### Related docs

- [`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md) — Tier A gaps + retune plan
- [`retune_campaign.md`](./retune_campaign.md) — wave / siege priority
- [`tf_alignment_audit.md`](./tf_alignment_audit.md) — TF ports; own-method split note
- [`accuracy_targets.md`](./accuracy_targets.md) — reference numbers
- [`fastmldnn_paper_comparison.md`](./fastmldnn_paper_comparison.md)
