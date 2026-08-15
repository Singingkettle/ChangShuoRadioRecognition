# CNN4 — Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels

> K. Youssef et al. / AMR-Benchmark CNN2 multipath, "Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels", *IEEE VTC (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/9128408](https://ieeexplore.ieee.org/abstract/document/9128408)

PyTorch / MMEngine port in CSRR. Algorithm short name **`cnn4`**
(= `configs/cnn4/`).

## Method in one paragraph

Multipath-oriented CNN (CSRR `CNN4`) with kernels fixed to (2,8) to match the AMR-Benchmark multipath CNN2 port.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/cnn4.py::CNN4` |
| Train / test configs | `configs/cnn4/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q |

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
python tools/train.py configs/cnn4/cnn4_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/cnn4/cnn4_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.57 / 58.00 | 83.89 / 80.00 | `pass` |
| RML2016.10B | 61.83 / 63.00 | 89.61 / 84.00 | `pass` |
| RML2018.01A | 54.55 / 55.00 | 84.57 / 91.00 | `fail` |
| HisarMod | 75.08 / 70.00 | 99.81 / 98.00 | `pass` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

Overall near-match on 2018; peak still short. SelfNormalize fine-tunes from official champs collapsed val and were abandoned.

