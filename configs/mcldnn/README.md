# MCLDNN — A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition

English | [简体中文](README_zh-CN.md)

> J. Xu, C. Yang, et al., "A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition", *IEEE Wireless Commun. Lett. (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/9106397](https://ieeexplore.ieee.org/abstract/document/9106397)

PyTorch / MMEngine port in CSRR. Algorithm short name **`mcldnn`**
(= `configs/mcldnn/`).

## Method in one paragraph

Multi-channel CNN + LSTM (MCLDNN). CSRR reshape matches Keras `(L-4, 100)`. Control model that passes all RML sets under 50/10/40.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/mcldnn.py::MCLDNN` |
| Train / test configs | `configs/mcldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q multi-branch |

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
python tools/train.py configs/mcldnn/mcldnn_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/mcldnn/mcldnn_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 61.75 / 62.00 | 92.45 / 92.05 | `pass` |
| RML2016.10B | 64.65 / 65.00 | 93.87 / 93.00 | `pass` |
| RML2018.01A | 61.56 / 60.00 | 96.83 / 95.00 | `pass` |
| HisarMod | 71.20 / 75.00 | 98.94 / 99.00 | `fail` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

Hisar overall still short; split already official. RML passes are the parity control for the port.

