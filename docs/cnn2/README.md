# CNN2 — Convolutional Radio Modulation Recognition Networks

> T. J. O’Shea, J. Corgan, and T. C. Clancy, "Convolutional Radio Modulation Recognition Networks", *EAI IntelliSys / Springer (2016)*.
> [https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16](https://link.springer.com/chapter/10.1007%2F978-3-319-44188-7_16)

PyTorch / MMEngine port in CSRR. Algorithm short name **`cnn2`**
(= `configs/cnn2/` = `docs/cnn2/`).

## Method in one paragraph

Classic O’Shea CNN1 (CSRR `CNN2`): two convolutional layers (50×1×8), dropout, and dense layers for 11-class RML2016.10A (and siblings for other datasets).

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/cnn2.py::CNN2` |
| Train / test configs | `configs/cnn2/` |
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
python tools/train.py configs/cnn2/cnn2_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/cnn2/cnn2_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 63.18 / 59.00 | 81.43 / 79.00 | `pass` |
| RML2016.10B | 56.25 / 64.00 | 81.58 / 85.00 | `fail` |
| RML2018.01A | 42.35 / 58.00 | 65.23 / 92.00 | `fail` |
| HisarMod | 79.74 / 75.00 | 100.00 / 100.00 | `pass` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

10A and Hisar pass. 10B and especially 2018 fail hard — 2018 long-sequence underperformance is structural relative to Fig. 5 readouts, not a missing pad.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
