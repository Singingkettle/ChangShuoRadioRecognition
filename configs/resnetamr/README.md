# ResNetAMR — Deep Neural Network Architectures for Modulation Classification (ResNet entry) / AMR-Benchmark ResNet

> X. Liu et al. / AMR-Benchmark ResNet, "Deep Neural Network Architectures for Modulation Classification (ResNet entry) / AMR-Benchmark ResNet", *IEEE Asilomar (2017) / DSP 2022 survey*.
> [https://ieeexplore.ieee.org/document/8335483](https://ieeexplore.ieee.org/document/8335483)

PyTorch / MMEngine port in CSRR. Algorithm short name **`resnetamr`**
(= `configs/resnetamr/`).

## Method in one paragraph

ResNet-style residual CNN for AMC as ported in AMR-Benchmark (`ResNetAMR`).

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/resnetamr.py::ResNetAMR` |
| Train / test configs | `configs/resnetamr/` |
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
python tools/train.py configs/resnetamr/resnetamr_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/resnetamr/resnetamr_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 55.95 / 57.00 | 84.14 / 83.00 | `pass` |
| RML2016.10B | 60.51 / 62.00 | 90.71 / 87.00 | `pass` |
| RML2018.01A | 57.10 / 57.00 | 93.53 / 91.00 | `pass` |
| HisarMod | 76.76 / 80.00 | 99.91 / 100.00 | `fail` |

Numbers are measured on the official `configs/` roots versus published / commonly cited targets (overall ≥ target−2.0 pp, peak ≥ target−1.5 pp).

## Documented deviations / notes

RML pass. Hisar overall short of approximate bar. Not previously listed in root README Supported Methods — added with this package.

