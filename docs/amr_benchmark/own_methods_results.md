# Project-own Methods — Measured vs Paper (MLDNN / HCGDNN / FastMLDNN)

This document tracks the reproduction of CSRR's three *own* methods
(MLDNN, HCGDNN, FastMLDNN) under the project-standard **50/10/40 split**
(`train.json` / `validation.json` / `test.json`) across all four
datasets, and compares the measured accuracy against the paper targets
documented in `accuracy_targets.md` (section "Project-own methods").

Pass rule is **one-sided**: a run passes when
`measured_overall >= target_overall - 1.5` **and**
`measured_peak >= target_peak - 1.0` (matching or exceeding the paper is
a pass). Best-SNR location is informational only (accuracy saturates on a
high-SNR plateau). Datasets a paper never reported are **measured-only**
(no pass/fail).

## Configuration audit & fixes (Phase: own-methods)

All configs were verified to use the standard 50/10/40 split and to
preserve each method's project-specific machinery. Fixes applied in this
workstream:

| Area | Before | After / fix |
|------|--------|-------------|
| MLDNN configs | already `MLDNNHead` (4 losses) + `MLDNNSNRLabel` + `MLDNNIQToAP`, standard split | verified; removed a duplicate `randomness` key in `configs/mldnn/schedules.py` that collided with `runtimes/amc.py` (would crash `Config.fromfile`) |
| HCGDNN 201610B / 201801A / hisar | used a plain `ClsHead` + shared dataset base (plain `Accuracy` val, no fusion) | switched to `HCGDNNHead` + local base + local `runtimes.py` (which carries `HCGDNNHook`) + local `schedules.py`; val metric set to `HCGDNNWeightsAccuracy` (fusion-weight learning) |
| HCGDNN hisar input shape | `[2, 1, 128]` (wrong length) | `[2, 1, 1024]` (HisarMod is 1024-long) |
| HCGDNN fusion weights | `register_buffer(..., persistent=False)` → not saved in checkpoint, so `tools/test.py` (no val pass) fell back to init weights (gru2=1) | `persistent=True` so the validation-learned fusion weights are saved into the best checkpoint and used by `test.py` for the fused `paper.pkl` |
| FastMLDNN 201610B / 201801A / hisar | plain `ClsHead` — would classify on the backbone's auxiliary `(x, p)` sim-matrix tuple during training (broken loss) | `FastMLDNNHead` (beta=0, matching the 201610A reference) preserving the central-distance-expansion loss machinery |
| `tools/test.py` collate | `Runner.build_dataloader` used mmengine default `pseudo_collate`; multi-input `{iq, ap}` models produced a list-of-dicts the data preprocessor cannot stack | default `collate_fn` to `default_collate` (matches `tools/train.py`); single-input baselines unaffected |

CPU smoke (`tools/amr_benchmark/_smoke_test.py`) passes for all 12
(method × dataset) configs: build + train-mode loss forward + test-mode
predict forward, with head dims matching dataset class counts
(11/10/24/26). The smoke harness was extended to handle multi-input dict
inputs and to exercise the real test pipeline for the predict path.

## Measured-vs-paper results

> Updated from `accuracy_tracking.md` (2026-07-14). Campaign mode for these
> three is **paper_exact**; **50/10/40 is paper-native** (not a TF concession).
> Deep dive: [`own_methods_paper_alignment.md`](./own_methods_paper_alignment.md).

| Method | Dataset | Target overall | Meas overall | Target peak | Meas peak | Peak SNR (meas) | Status | Notes |
|--------|---------|----------------|--------------|-------------|-----------|-----------------|--------|-------|
| MLDNN | RML2016.10A | 62.0 | **62.31** | 92.0 | **92.73** | 12 dB | **pass** (paper-exact) | Hold |
| MLDNN | RML2016.10B | n/a | 65.06 | n/a | 93.62 | 18 dB | measured | |
| MLDNN | RML2018.01A | n/a | 57.94 | n/a | 90.77 | 22 dB | measured | |
| MLDNN | HisarMod | n/a | 60.06 | n/a | 73.63 | 16 dB | measured | |
| HCGDNN | RML2016.10A | 64.9 | **63.04** | 93.0 | **93.11** | 18 dB | **fail** paper-exact | Peak OK; overall −1.86 |
| HCGDNN | RML2016.10B | n/a | 65.04 | n/a | 93.71 | 18 dB | measured | |
| HCGDNN | RML2018.01A | n/a | 58.72 | n/a | 93.52 | 24 dB | measured | |
| HCGDNN | HisarMod | n/a | 57.39 | n/a | 70.16 | 18 dB | measured | |
| FastMLDNN | RML2016.10A | 63.24 | **61.02** | 92.0 | **91.52** | 18 dB | **fail** paper-exact | Synced `esoff300`; paper siege next |
| FastMLDNN | RML2016.10B | n/a | 57.81 | n/a | 87.75 | 18 dB | measured | |
| FastMLDNN | RML2018.01A | n/a | 48.05 | n/a | 77.45 | 20 dB | measured | |
| FastMLDNN | HisarMod | n/a | 5.98 | n/a | 7.90 | −8 dB | measured | Broken; defer |

## Re-tuning log

_None yet. Bounded to ~3 substantive iterations per method. Primary lever
in reserve: per-sample L2 normalization on the IQ branch (the documented
RNN-ceiling root cause). Note `MLDNNIQToAP` does not normalize amplitude
(unlike `IQToAP`), so MLDNN/FastMLDNN IQ branches see the tiny native
signal scale — the candidate fix if their high-SNR accuracy falls short._

## Protocol differences vs each paper

- **Split (policy 2026-07-14).** Project standard **50/10/40 is treated as
  paper-native** for MLDNN / FastMLDNN / HCGDNN. Residual gaps must be closed
  via paper architecture freeze + training recipe (see alignment doc), not
  waived as “TF 6:2:2 vs CSRR.”
- **Datasets.** RML2016.10B, RML2018.01A and HisarMod have no extractable
  paper targets; measured-only.
