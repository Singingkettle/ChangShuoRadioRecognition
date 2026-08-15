# CLDNNW — Deep architectures for modulation recognition

> N. E. West and T. O’Shea, "Deep architectures for modulation recognition", *IEEE DySPAN (2017)*.
> [https://ieeexplore.ieee.org/abstract/document/7920754](https://ieeexplore.ieee.org/abstract/document/7920754)

PyTorch / MMEngine port in CSRR. Algorithm short name **`cldnnw`**
(= `configs/cldnnw/`).

## Method in one paragraph

West/O’Shea CLDNN: three (1×8) convolutions with dropout, concatenated features into an LSTM, then a dense classifier. CSRR restores TF `ZeroPadding2D((0,2))` before each conv via `use_zero_pad=True` (legacy no-pad checkpoints set it False).

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/cldnn.py::CLDNNW` |
| Train / test configs | `configs/cldnnw/` |
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
python tools/train.py configs/cldnnw/cldnnw_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/cldnnw/cldnnw_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 56.54 / 57.00 | 84.05 / 85.00 | `pass` |
| RML2016.10B | 60.35 / 62.00 | 88.05 / 89.00 | `pass` |
| RML2018.01A | 37.19 / 55.00 | 53.33 / 88.00 | `fail` |
| HisarMod | 66.54 / 75.00 | 96.17 / 98.00 | `fail` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

10A/10B pass under approximate tolerances after ZeroPad alignment. RML2018.01A and Hisar still fail by a wide margin (long-seq / platform). Do not re-siege identical wave17 Hisar loops.

