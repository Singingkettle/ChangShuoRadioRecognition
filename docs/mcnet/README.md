# MCNET — MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification

> T. Huynh-The et al., "MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification", *IEEE Commun. Lett. (2020)*.
> [https://ieeexplore.ieee.org/abstract/document/8963964](https://ieeexplore.ieee.org/abstract/document/8963964)

PyTorch / MMEngine port in CSRR. Algorithm short name **`mcnet`**
(= `configs/mcnet/` = `docs/mcnet/`).

## Method in one paragraph

Efficient CNN with M-blocks (MCNet) for robust AMC under channel impairments.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/mcnet.py::MCNet` |
| Train / test configs | `configs/mcnet/` |
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
python tools/train.py configs/mcnet/mcnet_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/mcnet/mcnet_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.04 / 58.00 | 84.59 / 82.00 | `pass` |
| RML2016.10B | 62.41 / 62.00 | 91.41 / 87.00 | `pass` |
| RML2018.01A | 58.43 / 55.00 | 92.78 / 92.00 | `pass` |
| HisarMod | 56.59 / 70.00 | 79.59 / 97.00 | `fail` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

RML pass. Hisar is a known hard case (DSP survey Table 4 highlights poor convergence); do not keep sieging L2+top1 wave17 clones.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
