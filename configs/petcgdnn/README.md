# PET-CGDNN — An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation

> F. Zhang et al., "An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation", *IEEE Commun. Lett. (2021)*.
> [https://ieeexplore.ieee.org/abstract/document/9507514](https://ieeexplore.ieee.org/abstract/document/9507514)

PyTorch / MMEngine port in CSRR. Algorithm short name **`petcgdnn`**
(= `configs/petcgdnn/`).

## Method in one paragraph

Parameter-estimation transform (PET) rotates I/Q before a compact CGDNN classifier. Q-rotation sign matches TF.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/petcgdnn.py::PETCGDNN` |
| Train / test configs | `configs/petcgdnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q + PET rotation |

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
python tools/train.py configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/petcgdnn/petcgdnn_iq-shape-L-F-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 60.26 / 60.00 | 90.43 / 89.00 | `pass` |
| RML2016.10B | 63.80 / 63.00 | 93.52 / 92.00 | `pass` |
| RML2018.01A | 61.24 / 60.00 | 95.69 / 95.00 | `pass` |
| HisarMod | 67.35 / 75.00 | 90.68 / 99.00 | `fail` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

RML pass. Hisar fails; split already official.

