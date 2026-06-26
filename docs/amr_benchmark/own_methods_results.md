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

> _Status: training pending — coordinating with the running AMR-Benchmark
> sweep (GPUs busy). The durable sweep driver's RML2018.01A / HisarMod
> phases enumerate these three methods automatically (they are in
> `tools/amr_benchmark/matrix.py`); RML2016.10A/10B are produced during
> the same phased sweep. Numbers will be filled in here as `paper.pkl`
> files land under `work_dirs/amr_benchmark/<model>/<dataset>/res/`._

| Method | Dataset | Target overall | Meas overall | Target peak | Meas peak | Peak SNR (meas) | Status | Notes |
|--------|---------|----------------|--------------|-------------|-----------|-----------------|--------|-------|
| MLDNN | RML2016.10A | ~62.0 (fig) | — | ~92.0 | — | — | pending | approx target; 50/10/40 vs paper protocol |
| MLDNN | RML2016.10B | n/a | — | n/a | — | — | pending | measured-only |
| MLDNN | RML2018.01A | n/a | — | n/a | — | — | pending | measured-only |
| MLDNN | HisarMod | n/a | — | n/a | — | — | pending | measured-only |
| HCGDNN | RML2016.10A | 64.9 | — | ~93.0 | — | — | pending | fused (HCGDNNWeightsAccuracy) |
| HCGDNN | RML2016.10B | n/a | — | n/a | — | — | pending | measured-only |
| HCGDNN | RML2018.01A | n/a | — | n/a | — | — | pending | measured-only |
| HCGDNN | HisarMod | n/a | — | n/a | — | — | pending | measured-only |
| FastMLDNN | RML2016.10A | 63.24 | — | ~92.0 | — | — | pending | FastMLDNNHead beta=0 |
| FastMLDNN | RML2016.10B | n/a | — | n/a | — | — | pending | measured-only |
| FastMLDNN | RML2018.01A | n/a | — | n/a | — | — | pending | measured-only |
| FastMLDNN | HisarMod | n/a | — | n/a | — | — | pending | measured-only |

## Re-tuning log

_None yet. Bounded to ~3 substantive iterations per method. Primary lever
in reserve: per-sample L2 normalization on the IQ branch (the documented
RNN-ceiling root cause). Note `MLDNNIQToAP` does not normalize amplitude
(unlike `IQToAP`), so MLDNN/FastMLDNN IQ branches see the tiny native
signal scale — the candidate fix if their high-SNR accuracy falls short._

## Protocol differences vs each paper

- **Split.** All three papers report on RML2016.10A with their own
  protocol; MLDNN/FastMLDNN historically merged train+validation and
  re-used the test set for validation. This workstream uses the strict
  50/10/40 split, so a small (~1–3 pp) absolute drop vs the published
  numbers is expected and is reported as a documented gap, not a failure,
  when within reason.
- **Datasets.** RML2016.10B, RML2018.01A and HisarMod were not given
  explicit accuracy numbers we could extract from the three papers; they
  are trained and reported measured-only.
