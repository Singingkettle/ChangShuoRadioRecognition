# IC-AMCNet — CNN-Based Automatic Modulation Classification for Beyond 5G Communications

> A. P. Hermawan et al., "CNN-Based Automatic Modulation Classification for Beyond 5G Communications", *IEEE Commun. Lett. (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/8977561](https://ieeexplore.ieee.org/abstract/document/8977561)

PyTorch / MMEngine port in CSRR. Algorithm short name **`icamcnet`**
(= `configs/icamcnet/` = `docs/icamcnet/`).

## Method in one paragraph

Deep CNN with Gaussian noise regularization (IC-AMCNet). Large parameter count on long frames (2018/Hisar).

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/icamcnet.py::ICAMCNet` |
| Train / test configs | `configs/icamcnet/` |
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
python tools/train.py configs/icamcnet/icamcnet_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/icamcnet/icamcnet_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 56.79 / 57.00 | 85.07 / 83.00 | `pass` |
| RML2016.10B | 61.66 / 62.00 | 91.67 / 87.00 | `pass` |
| RML2018.01A | 59.49 / 58.00 | 95.13 / 92.00 | `pass` |
| HisarMod | 83.41 / 80.00 | 98.58 / 100.00 | `pass` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

All four tracking datasets pass (Hisar peak via near-match 98.58 ≥ 98.5). Peak-100 ES loops are closed.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
