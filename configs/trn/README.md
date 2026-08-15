# TRN — Signal Modulation Classification Based on the Transformer Network

English | [简体中文](README_zh-CN.md)

> J. Cai et al., "Signal Modulation Classification Based on the Transformer Network", *IEEE Trans. Cogn. Commun. Netw. (2022)*.
> [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779340)

PyTorch / MMEngine port in CSRR. Algorithm short name **`trn`**
(= `configs/trn/`).

## Method in one paragraph

Transformer network on constellation / image-like inputs for modulation classification.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `(image-backbone path via configs/trn)` |
| Train / test configs | `configs/trn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | Image / constellation |

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
python tools/train.py configs/trn/trn_img-deepsig201610A.py

# test a checkpoint
python tools/test.py configs/trn/trn_img-deepsig201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

_No AMR-Benchmark tracking row for this method. Use the train command below and record metrics locally._

## Documented deviations / notes

Not in the closed AMR I/Q tracking matrix; image pipeline configs live under `configs/trn/`.

