# MLDNN — Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification

> S. Chang et al., "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification", *IEEE Trans. Veh. Technol. (2021)*.
> [https://ieeexplore.ieee.org/document/9462447](https://ieeexplore.ieee.org/document/9462447)

PyTorch / MMEngine port in CSRR. Algorithm short name **`mldnn`**
(= `configs/mldnn/` = `docs/mldnn/`).

## Method in one paragraph

Own-method Tier A: multitask MLDNN with shared trunk and modulation (+ optional SNR) heads. Paper-native 50/10/40.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/mldnn.py::MLDNN` |
| Train / test configs | `configs/mldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q + A/P |

## Data

DeepSig RML JSON under `data/ModulationClassification/DeepSig/` uses CSRR
**50/10/40** (`train.json` / `validation.json` / `test.json`). TF AMR-Benchmark
RML used per-(modulation, SNR) **6:2:2**; small overall gaps on Tier-B ports may
cite that difference.

HisarMod live JSON under `data/ModulationClassification/Hisar/HisarMod2019.1/`
already follows the **official Test + Train 80/20** protocol
(~416k / 104k / 260k). Do not attribute Hisar residuals to a 50/10/40 Hisar split.

## Train / evaluate

```bash
# train (default work_dir under work_dirs/)
python tools/train.py configs/mldnn/mldnn_iq-ap-deepsig201610A.py

# test a checkpoint
python tools/test.py configs/mldnn/mldnn_iq-ap-deepsig201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 62.31 / 62.00 | 92.73 / 92.00 | `pass` |
| RML2016.10B | 65.06 / (CSRR-only) | 93.62 / (CSRR-only) | `measured` |
| RML2018.01A | 57.94 / (CSRR-only) | 90.77 / (CSRR-only) | `measured` |
| HisarMod | 60.06 / (CSRR-only) | 73.63 / (CSRR-only) | `measured` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

10A is paper-exact pass. Other datasets measured-only under CSRR.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
