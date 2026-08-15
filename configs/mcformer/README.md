# MCformer — MCformer: A Transformer Based Deep Neural Network for Automatic Modulation Classification

> S. Hamidi-Rad and S. Jain, "MCformer: A Transformer Based Deep Neural Network for Automatic Modulation Classification", *IEEE Commun. Lett. (2022)*.
> [https://ieeexplore.ieee.org/abstract/document/9685815](https://ieeexplore.ieee.org/abstract/document/9685815)

PyTorch / MMEngine port in CSRR. Algorithm short name **`mcformer`**
(= `configs/mcformer/`).

## Method in one paragraph

Transformer encoder for AMC on reshaped I/Q patches.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/mcformer.py::MCformer` |
| Train / test configs | `configs/mcformer/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | I/Q (F×L) |

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
python tools/train.py configs/mcformer/mcformer_iq-shape-F-L-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/mcformer/mcformer_iq-shape-F-L-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

_No AMR-Benchmark tracking row for this method. Use the train command below and record metrics locally._

## Documented deviations / notes

Not in the closed AMR tracking matrix; configs provided for community use.

