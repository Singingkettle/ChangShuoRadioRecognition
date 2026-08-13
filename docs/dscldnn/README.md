# DSCLDNN — Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure

> Z. Zhang et al., "Automatic Modulation Classification Using CNN-LSTM Based Dual-Stream Structure", *IEEE Access (2020)*.
> [https://ieeexplore.ieee.org/document/9220797](https://ieeexplore.ieee.org/document/9220797)

PyTorch / MMEngine port in CSRR. Algorithm short name **`dscldnn`**
(= `configs/dscldnn/` = `docs/dscldnn/`).

## Method in one paragraph

Dual-stream CNN–LSTM: one stream on I/Q and one on amplitude/phase, fused before classification.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/dscldnn.py::DSCLDNN` |
| Train / test configs | `configs/dscldnn/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P + I/Q dual stream |

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
python tools/train.py configs/dscldnn/dscldnn_ap-iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/dscldnn/dscldnn_ap-iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

_No AMR-Benchmark tracking row for this method. Use the train command below and record metrics locally._

## Documented deviations / notes

Not part of the closed AMR-Benchmark tracking matrix; configs are provided for completeness. Use the root configs under `configs/dscldnn/`.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
