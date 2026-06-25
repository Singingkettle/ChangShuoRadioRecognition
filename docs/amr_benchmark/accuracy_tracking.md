# AMR-Benchmark Accuracy Tracking

This file is the live results table that the orchestrator
(`tools/amr_benchmark/run_migration.py`) updates after every
`train.py → test.py` pair. Reference numbers come from
[`accuracy_targets.md`](./accuracy_targets.md); pass/fail is
computed against the tolerance bands documented there.

| Status legend |
|---------------|
| `pending` — not yet trained |
| `running` — training/eval in progress |
| `pass`    — measured ≤ tolerance away from target |
| `fail`    — measured outside tolerance |
| `error`   — train/test pipeline failed |

## Tolerances

- overall_acc tolerance: **±1.5 pp**
- peak_acc tolerance:    **±1.0 pp**
- peak_snr tolerance:    **±2 dB**

## Results

| Model | Dataset | Config | Work dir | Overall (target) | Overall (meas) | Peak (target) | Peak (meas) | Best SNR (target) | Best SNR (meas) | Status | Updated |
|-------|---------|--------|----------|------------------|-----------------|---------------|--------------|--------------------|------------------|--------|---------|
| CNN2 | RML2016.10A | configs/cnn2/cnn2_iq-deepsig-201610A.py | (TBD) | ~59% | — | ~79% | — | ≥6 dB | — | pending | — |
| CNN2 | RML2016.10B | configs/cnn2/cnn2_iq-deepsig-201610B.py | (TBD) | ~64% | — | ~85% | — | ≥4 dB | — | pending | — |
| CNN2 | RML2018.01A | configs/cnn2/cnn2_iq-deepsig-201801A.py | (TBD) | ~58% | — | ~92% | — | ≥18 dB | — | pending | — |
| CNN2 | HisarMod | configs/cnn2/cnn2_iq-hisar-2019.py | (TBD) | ~75% | — | ~100% | — | ≥10 dB | — | pending | — |
| CNN4 | RML2016.10A | configs/cnn4/cnn4_iq-deepsig-201610A.py | (TBD) | ~58% | — | ~80% | — | ≥4 dB | — | pending | — |
| CNN4 | RML2016.10B | configs/cnn4/cnn4_iq-deepsig-201610B.py | (TBD) | ~63% | — | ~84% | — | ≥2 dB | — | pending | — |
| CNN4 | RML2018.01A | configs/cnn4/cnn4_iq-deepsig-201801A.py | (TBD) | ~55% | — | ~91% | — | ≥18 dB | — | pending | — |
| CNN4 | HisarMod | configs/cnn4/cnn4_iq-hisar-2019.py | (TBD) | ~70% | — | ~98% | — | ≥10 dB | — | pending | — |
| MCNet | RML2016.10A | configs/mcnet/mcnet_iq-deepsig-201610A.py | (TBD) | ~58% | — | ~82% | — | ≥6 dB | — | pending | — |
| MCNet | RML2016.10B | configs/mcnet/mcnet_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~87% | — | ≥4 dB | — | pending | — |
| MCNet | RML2018.01A | configs/mcnet/mcnet_iq-deepsig-201801A.py | (TBD) | ~55% | — | ~92% | — | ≥18 dB | — | pending | — |
| MCNet | HisarMod | configs/mcnet/mcnet_iq-hisar-2019.py | (TBD) | ~70% | — | ~97% | — | ≥10 dB | — | pending | — |
| ICAMCNet | RML2016.10A | configs/icamcnet/icamcnet_iq-deepsig-201610A.py | (TBD) | ~57% | — | ~83% | — | ≥6 dB | — | pending | — |
| ICAMCNet | RML2016.10B | configs/icamcnet/icamcnet_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~87% | — | ≥4 dB | — | pending | — |
| ICAMCNet | RML2018.01A | configs/icamcnet/icamcnet_iq-deepsig-201801A.py | (TBD) | ~58% | — | ~92% | — | ≥18 dB | — | pending | — |
| ICAMCNet | HisarMod | configs/icamcnet/icamcnet_iq-hisar-2019.py | (TBD) | ~80% | — | ~100% | — | ≥10 dB | — | pending | — |
| ResNetAMR | RML2016.10A | configs/resnetamr/resnetamr_iq-deepsig-201610A.py | (TBD) | ~57% | — | ~83% | — | ≥6 dB | — | pending | — |
| ResNetAMR | RML2016.10B | configs/resnetamr/resnetamr_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~87% | — | ≥4 dB | — | pending | — |
| ResNetAMR | RML2018.01A | configs/resnetamr/resnetamr_iq-deepsig-201801A.py | (TBD) | ~57% | — | ~91% | — | ≥18 dB | — | pending | — |
| ResNetAMR | HisarMod | configs/resnetamr/resnetamr_iq-hisar-2019.py | (TBD) | ~80% | — | ~100% | — | ≥10 dB | — | pending | — |
| DensCNN | RML2016.10A | configs/denscnn/denscnn_iq-deepsig-201610A.py | (TBD) | ~57% | — | ~83% | — | ≥6 dB | — | pending | — |
| DensCNN | RML2016.10B | configs/denscnn/denscnn_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~87% | — | ≥4 dB | — | pending | — |
| DensCNN | RML2018.01A | configs/denscnn/denscnn_iq-deepsig-201801A.py | (TBD) | ~58% | — | ~92% | — | ≥18 dB | — | pending | — |
| DensCNN | HisarMod | configs/denscnn/denscnn_iq-hisar-2019.py | (TBD) | ~80% | — | ~100% | — | ≥10 dB | — | pending | — |
| GRU2 | RML2016.10A | configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py | (TBD) | ~58% | — | ~85% | — | ≥4 dB | — | pending | — |
| GRU2 | RML2016.10B | configs/gru2/gru2_iq-shape-L-F-deepsig-201610B.py | (TBD) | ~63% | — | ~91% | — | ≥2 dB | — | pending | — |
| GRU2 | RML2018.01A | configs/gru2/gru2_iq-shape-L-F-deepsig-201801A.py | (TBD) | ~59% | — | ~95% | — | ≥18 dB | — | pending | — |
| GRU2 | HisarMod | configs/gru2/gru2_iq-shape-L-F-hisar-2019.py | (TBD) | ~73% | — | ~98% | — | ≥10 dB | — | pending | — |
| LSTM2 | RML2016.10A | configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py | (TBD) | ~58% | — | ~87% | — | ≥4 dB | — | pending | — |
| LSTM2 | RML2016.10B | configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610B.py | (TBD) | ~64% | — | ~94% | — | 18 dB | — | pending | — |
| LSTM2 | RML2018.01A | configs/lstm2/lstm2_ap-shape-L-F-deepsig-201801A.py | (TBD) | ~60% | — | ~98% | — | 22 dB | — | pending | — |
| LSTM2 | HisarMod | configs/lstm2/lstm2_ap-shape-L-F-hisar-2019.py | (TBD) | ~73% | — | ~98% | — | ≥10 dB | — | pending | — |
| DAE | RML2016.10A | configs/dae/dae_ap-deepsig-201610A.py | (TBD) | ~57% | — | ~82% | — | ≥6 dB | — | pending | — |
| DAE | RML2016.10B | configs/dae/dae_ap-deepsig-201610B.py | (TBD) | ~62% | — | ~85% | — | ≥4 dB | — | pending | — |
| DAE | RML2018.01A | configs/dae/dae_ap-deepsig-201801A.py | (TBD) | ~55% | — | ~90% | — | ≥18 dB | — | pending | — |
| DAE | HisarMod | configs/dae/dae_ap-hisar-2019.py | (TBD) | ~40% | — | ~70% | — | ≥10 dB | — | pending | — |
| MCLDNN | RML2016.10A | configs/mcldnn/mcldnn_iq-deepsig-201610A.py | (TBD) | ~62% | — | **92.05%** | — | **10 dB** | — | pending | — |
| MCLDNN | RML2016.10B | configs/mcldnn/mcldnn_iq-deepsig-201610B.py | (TBD) | ~65% | — | ~93% | — | ≥4 dB | — | pending | — |
| MCLDNN | RML2018.01A | configs/mcldnn/mcldnn_iq-deepsig-201801A.py | (TBD) | ~60% | — | ~95% | — | ≥18 dB | — | pending | — |
| MCLDNN | HisarMod | configs/mcldnn/mcldnn_iq-hisar-2019.py | (TBD) | ~75% | — | ~99% | — | ≥10 dB | — | pending | — |
| CLDNNW | RML2016.10A | configs/cldnnw/cldnnw_iq-deepsig-201610A.py | (TBD) | ~57% | — | ~85% | — | ≥6 dB | — | pending | — |
| CLDNNW | RML2016.10B | configs/cldnnw/cldnnw_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~89% | — | ≥4 dB | — | pending | — |
| CLDNNW | RML2018.01A | configs/cldnnw/cldnnw_iq-deepsig-201801A.py | (TBD) | ~55% | — | ~88% | — | ≥18 dB | — | pending | — |
| CLDNNW | HisarMod | configs/cldnnw/cldnnw_iq-hisar-2019.py | (TBD) | ~75% | — | ~98% | — | ≥10 dB | — | pending | — |
| CLDNNL | RML2016.10A | configs/cldnnl/cldnnl_iq-deepsig-201610A.py | (TBD) | ~57% | — | ~85% | — | ≥4 dB | — | pending | — |
| CLDNNL | RML2016.10B | configs/cldnnl/cldnnl_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~89% | — | ≥2 dB | — | pending | — |
| CLDNNL | RML2018.01A | configs/cldnnl/cldnnl_iq-deepsig-201801A.py | (TBD) | ~57% | — | ~92% | — | ≥18 dB | — | pending | — |
| CLDNNL | HisarMod | configs/cldnnl/cldnnl_iq-hisar-2019.py | (TBD) | ~75% | — | ~98% | — | ≥10 dB | — | pending | — |
| CGDNet | RML2016.10A | configs/cgdnet/cgdnet_iq-deepsig-201610A.py | (TBD) | ~58% | — | ~83% | — | ≥6 dB | — | pending | — |
| CGDNet | RML2016.10B | configs/cgdnet/cgdnet_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~88% | — | ≥4 dB | — | pending | — |
| CGDNet | RML2018.01A | configs/cgdnet/cgdnet_iq-deepsig-201801A.py | (TBD) | ~57% | — | ~92% | — | ≥18 dB | — | pending | — |
| CGDNet | HisarMod | configs/cgdnet/cgdnet_iq-hisar-2019.py | (TBD) | (CSRR-only) | — | (CSRR-only) | — | ≥10 dB | — | pending | — |
| PETCGDNN | RML2016.10A | configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py | (TBD) | ~60% | — | ~89% | — | ≥6 dB | — | pending | — |
| PETCGDNN | RML2016.10B | configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610B.py | (TBD) | ~63% | — | ~92% | — | ≥4 dB | — | pending | — |
| PETCGDNN | RML2018.01A | configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201801A.py | (TBD) | ~60% | — | ~95% | — | ≥18 dB | — | pending | — |
| PETCGDNN | HisarMod | configs/petcgdnn/petcgdnn_iq-shape-L-F-hisar-2019.py | (TBD) | ~75% | — | ~99% | — | ≥10 dB | — | pending | — |
| CNN1DPF | RML2016.10A | configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py | (TBD) | ~57% | — | ~85% | — | ≥6 dB | — | pending | — |
| CNN1DPF | RML2016.10B | configs/cnn1dpf/cnn1dpf_iq-deepsig-201610B.py | (TBD) | ~62% | — | ~88% | — | ≥4 dB | — | pending | — |
| CNN1DPF | RML2018.01A | configs/cnn1dpf/cnn1dpf_iq-deepsig-201801A.py | (TBD) | ~57% | — | ~91% | — | ≥18 dB | — | pending | — |
| CNN1DPF | HisarMod | configs/cnn1dpf/cnn1dpf_iq-hisar-2019.py | (TBD) | (CSRR-only) | — | (CSRR-only) | — | ≥10 dB | — | pending | — |

> **Last full update:** _Pending Phase 2._ Phase 1A leaves all
> measured columns empty; the orchestrator will populate them
> automatically after each (model × dataset) finishes.

## Auto-generated results

The block between the markers below is rewritten by
`tools/amr_benchmark/run_migration.py` on every successful job.
Do not hand-edit the rows; edit the targets in
`tools/amr_benchmark/matrix.py` (or this file's manual table above)
instead.

<!-- AMR_BENCHMARK_AUTO_TABLE_BEGIN -->
_Last orchestrator run: 2026-06-25 21:29:54 UTC_

| Model | Dataset | Config | Work dir | Overall (target %) | Overall (meas %) | Peak (target %) | Peak (meas %) | Best SNR (target) | Best SNR (meas) | Status | Updated |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN2 | RML2016.10A | `configs/cnn2/cnn2_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/cnn2/deepsig201610A` | 59.00 | 63.60 | 79.00 | 81.70 | 6 dB | 10 dB | `fail` | 2026-06-25T21:07:34Z |
| CNN4 | RML2016.10A | `configs/cnn4/cnn4_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/cnn4/deepsig201610A` | 58.00 | 57.57 | 80.00 | 83.89 | 4 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| MCNET | RML2016.10A | `configs/mcnet/mcnet_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/mcnet/deepsig201610A` | 58.00 | 55.98 | 82.00 | 82.41 | 6 dB | 12 dB | `fail` | 2026-06-25T21:07:34Z |
| ICAMCNET | RML2016.10A | `configs/icamcnet/icamcnet_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/icamcnet/deepsig201610A` | 57.00 | 56.79 | 83.00 | 85.07 | 6 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| RESNETAMR | RML2016.10A | `configs/resnetamr/resnetamr_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/resnetamr/deepsig201610A` | 57.00 | 54.33 | 83.00 | 82.20 | 6 dB | 10 dB | `fail` | 2026-06-25T21:07:34Z |
| DENSCNN | RML2016.10A | `configs/denscnn/denscnn_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/denscnn/deepsig201610A` | 57.00 | 54.48 | 83.00 | 82.86 | 6 dB | 10 dB | `fail` | 2026-06-25T21:07:34Z |
| GRU2 | RML2016.10A | `configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py` | `work_dirs/amr_benchmark/gru2/deepsig201610A` | 58.00 | 57.74 | 85.00 | 85.70 | 4 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| LSTM2 | RML2016.10A | `configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py` | `work_dirs/amr_benchmark/lstm2/deepsig201610A` | 58.00 | 56.57 | 87.00 | 85.11 | 4 dB | 12 dB | `fail` | 2026-06-25T21:07:34Z |
| DAE | RML2016.10A | `configs/dae/dae_ap-deepsig-201610A.py` | `work_dirs/amr_benchmark/dae/deepsig201610A` | 57.00 | 55.60 | 82.00 | 84.68 | 6 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| MCLDNN | RML2016.10A | `configs/mcldnn/mcldnn_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/mcldnn/deepsig201610A` | 62.00 | 57.81 | 92.05 | 85.23 | 10 dB | 18 dB | `fail` | 2026-06-25T21:29:54Z |
| CLDNNW | RML2016.10A | `configs/cldnnw/cldnnw_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/cldnnw/deepsig201610A` | 57.00 | 53.00 | 85.00 | 78.25 | 6 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| CLDNNL | RML2016.10A | `configs/cldnnl/cldnnl_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/cldnnl/deepsig201610A` | 57.00 | 57.53 | 85.00 | 83.80 | 4 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| CGDNET | RML2016.10A | `configs/cgdnet/cgdnet_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/cgdnet/deepsig201610A` | 58.00 | 53.46 | 83.00 | 79.18 | 6 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
| PETCGDNN | RML2016.10A | `configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py` | `work_dirs/amr_benchmark/petcgdnn/deepsig201610A` | 60.00 | 57.90 | 89.00 | 86.45 | 6 dB | 14 dB | `fail` | 2026-06-25T21:07:34Z |
| CNN1DPF | RML2016.10A | `configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py` | `work_dirs/amr_benchmark/cnn1dpf/deepsig201610A` | 57.00 | 54.97 | 85.00 | 84.16 | 6 dB | 18 dB | `fail` | 2026-06-25T21:07:34Z |
<!-- AMR_BENCHMARK_AUTO_TABLE_END -->
