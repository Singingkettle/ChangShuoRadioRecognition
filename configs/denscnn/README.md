# DensCNN — Deep Neural Network Architectures for Modulation Classification

> X. Liu, D. Yang, and A. El Gamal, "Deep Neural Network Architectures for Modulation Classification", *IEEE Asilomar (2017)*.
> [https://ieeexplore.ieee.org/document/8335483](https://ieeexplore.ieee.org/document/8335483)

PyTorch / MMEngine port in CSRR. Algorithm short name **`denscnn`**
(= `configs/denscnn/`).

## Method in one paragraph

DenseNet-style convolutional classifier from the Asilomar architecture paper / AMR-Benchmark DenseNet entry (`DensCNN`).

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/denscnn.py::DensCNN` |
| Train / test configs | `configs/denscnn/` |
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
python tools/train.py configs/denscnn/denscnn_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/denscnn/denscnn_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 58.92 / 57.00 | 90.30 / 83.00 | `pass` |
| RML2016.10B | 60.72 / 62.00 | 91.87 / 87.00 | `pass` |
| RML2018.01A | 53.99 / 58.00 | 89.45 / 92.00 | `fail` |
| HisarMod | 83.85 / 80.00 | 100.00 / 100.00 | `pass` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

10A/10B/Hisar pass. 2018 overall/peak sit just under the approximate bars; further SelfNormalize FT regressed.

