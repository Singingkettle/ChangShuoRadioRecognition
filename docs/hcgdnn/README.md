# HCGDNN — A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification

> S. Chang et al., "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification", *IEEE Wireless Commun. Lett. (2022)*.
> [https://ieeexplore.ieee.org/document/9764618](https://ieeexplore.ieee.org/document/9764618)

PyTorch / MMEngine port in CSRR. Algorithm short name **`hcgdnn`**
(= `configs/hcgdnn/` = `docs/hcgdnn/`).

## Method in one paragraph

Own-method Tier A: hierarchical classification head on a convolutional gated network. Paper-native 50/10/40.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/hcgdnn.py::HCGDNN` |
| Train / test configs | `configs/hcgdnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q |

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
python tools/train.py configs/hcgdnn/hcgdnn_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/hcgdnn/hcgdnn_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 63.43 / 64.90 | 93.36 / 93.00 | `pass` |
| RML2016.10B | 65.04 / (CSRR-only) | 93.71 / (CSRR-only) | `measured` |
| RML2018.01A | 58.72 / (CSRR-only) | 93.52 / (CSRR-only) | `measured` |
| HisarMod | 57.39 / (CSRR-only) | 70.16 / (CSRR-only) | `measured` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

10A tracking pass 63.43/93.36 vs 64.9/93. Other datasets measured-only. Paper-exact sieges closed.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
