# CLDNNL — Deep Neural Network Architectures for Modulation Classification

> X. Liu, D. Yang, and A. El Gamal, "Deep Neural Network Architectures for Modulation Classification", *IEEE Asilomar (2017)*.
> [https://ieeexplore.ieee.org/document/8335483](https://ieeexplore.ieee.org/document/8335483)

PyTorch / MMEngine port in CSRR. Algorithm short name **`cldnnl`**
(= `configs/cldnnl/` = `docs/cldnnl/`).

## Method in one paragraph

Liu/Yang/El Gamal CLDNN2-style stack (CSRR name `CLDNNL`): CNN front-end plus LSTM aggregator, matching the AMR-Benchmark CLDNN2 Keras recipe.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/cldnn.py::CLDNNL` |
| Train / test configs | `configs/cldnnl/` |
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
python tools/train.py configs/cldnnl/cldnnl_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/cldnnl/cldnnl_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 61.23 / 57.00 | 90.91 / 85.00 | `pass` |
| RML2016.10B | 63.63 / 62.00 | 92.73 / 89.00 | `pass` |
| RML2018.01A | 46.67 / 57.00 | 81.40 / 92.00 | `fail` |
| HisarMod | 70.26 / 75.00 | 89.47 / 98.00 | `fail` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

10A/10B pass. 2018 and Hisar remain fails; remaining gaps are treated as approximate-mode ceilings, not missing layers.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
