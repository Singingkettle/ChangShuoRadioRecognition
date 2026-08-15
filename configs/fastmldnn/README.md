# FastMLDNN — A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification

English | [简体中文](README_zh-CN.md)

> S. Chang et al., "A Fast Multi-Loss Learning Deep Neural Network for Automatic Modulation Classification", *IEEE Trans. Cogn. Commun. Netw. (2023)*.
> [https://ieeexplore.ieee.org/abstract/document/10239249](https://ieeexplore.ieee.org/abstract/document/10239249)

PyTorch / MMEngine port in CSRR. Algorithm short name **`fastmldnn`**
(= `configs/fastmldnn/`).

## Method in one paragraph

Own-method Tier A: multi-loss FastMLDNN with I/Q and A/P branches. Paper-native split is 50/10/40 — do not attribute residuals to TF 6:2:2.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/fastmldnn.py::FastMLDNN` |
| Train / test configs | `configs/fastmldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q + A/P |

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
python tools/train.py configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 61.42 / 63.24 | 92.98 / 92.00 | `pass` |
| RML2016.10B | 57.81 / (CSRR-only) | 87.75 / (CSRR-only) | `measured` |
| RML2018.01A | 48.05 / (CSRR-only) | 77.45 / (CSRR-only) | `measured` |
| HisarMod | 5.98 / (CSRR-only) | 7.90 / (CSRR-only) | `measured` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

10A tracking pass at 61.42/92.98 vs paper 63.24/92 (approximate/near-match). Other datasets are measured-only; Hisar default run is broken (~6%) and not a reproduction claim. Further paper-exact seed/FT sieges are closed.

