# GRU2 — Automatic modulation classification using recurrent neural networks

> D. Hong et al. / AMR-Benchmark GRU, "Automatic modulation classification using recurrent neural networks", *IEEE ICSPCC (2017)*.
> [https://ieeexplore.ieee.org/abstract/document/8322633](https://ieeexplore.ieee.org/abstract/document/8322633)

PyTorch / MMEngine port in CSRR. Algorithm short name **`gru2`**
(= `configs/gru2/`).

## Method in one paragraph

Two-layer GRU classifier on reshaped I/Q (`L×F`). Matches the AMR-Benchmark GRU Keras path used in the DSP survey.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/gru2.py::GRU2` |
| Train / test configs | `configs/gru2/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q (L×F reshape) |

## Data

DeepSig RML JSON under `data/ModulationClassification/DeepSig/` uses CSRR
**50/10/40** (`train.json` / `validation.json` / `test.json`). Some public Keras
ports use per-(modulation, SNR) **6:2:2**; small overall gaps on a few datasets may
cite that difference.

HisarMod live JSON under `data/ModulationClassification/Hisar/HisarMod2019.1/`
already follows the **official Test + Train 80/20** protocol
(~416k / 104k / 260k). Do not attribute Hisar residuals to a 50/10/40 Hisar split.

## Train / evaluate

```bash
# train (default work_dir under work_dirs/)
python tools/train.py configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/gru2/gru2_iq-shape-L-F-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.84 / 58.00 | 85.95 / 85.00 | `pass` |
| RML2016.10B | 64.53 / 63.00 | 93.50 / 91.00 | `pass` |
| RML2018.01A | 61.95 / 59.00 | 96.37 / 95.00 | `pass` |
| HisarMod | 69.34 / 73.00 | 97.02 / 98.00 | `fail` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

All RML pass. Hisar overall short of the approximate bar after wave17 plateau; split is already official Test + Train 80/20.

