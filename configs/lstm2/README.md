# LSTM2 — Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors

> S. Rajendran et al., "Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors", *IEEE Trans. Cogn. Commun. Netw. (2018)*.
> [https://ieeexplore.ieee.org/abstract/document/8357902](https://ieeexplore.ieee.org/abstract/document/8357902)

PyTorch / MMEngine port in CSRR. Algorithm short name **`lstm2`**
(= `configs/lstm2/`).

## Method in one paragraph

Two-layer LSTM on amplitude/phase (L×F). TF and CSRR both use A/P — raw I/Q collapses accuracy.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/lstm2.py::LSTM2` |
| Train / test configs | `configs/lstm2/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P |

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
python tools/train.py configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/lstm2/lstm2_ap-shape-L-F-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 63.53 / 58.00 | 93.73 / 87.00 | `pass` |
| RML2016.10B | 63.94 / 64.00 | 93.66 / 94.00 | `pass` |
| RML2018.01A | 62.30 / 60.00 | 97.02 / 98.00 | `pass` |
| HisarMod | 69.91 / 73.00 | 97.00 / 98.00 | `fail` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

RML pass. Hisar overall short after polish; Hisar split already official.

