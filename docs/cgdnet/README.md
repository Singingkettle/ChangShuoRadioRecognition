# CGDNet — CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition

> Y. Wang et al., "CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition", *IEEE Commun. Lett.* (2021).
> [https://ieeexplore.ieee.org/abstract/document/9349627](https://ieeexplore.ieee.org/abstract/document/9349627)

PyTorch / MMEngine port in CSRR. Algorithm short name **`cgdnet`**
(= `configs/cgdnet/` = `docs/cgdnet/`).

## Method in one paragraph

A compact CNN–GRU hybrid: convolutional front-end extracts local I/Q features, then a gated recurrent unit aggregates temporal context for modulation class prediction. CSRR ports the AMR-Benchmark CGDNet topology with `frame_length` fixed for long sequences (2018 / Hisar).

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/cgdnet.py::CGDNet` |
| Train / test configs | `configs/cgdnet/` |
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
python tools/train.py configs/cgdnet/cgdnet_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/cgdnet/cgdnet_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 56.96 / 58.00 | 83.66 / 83.00 | `pass` |
| RML2016.10B | 61.15 / 62.00 | 89.49 / 88.00 | `pass` |
| RML2018.01A | 35.87 / 57.00 | 51.67 / 92.00 | `fail` |
| HisarMod | 71.25 / (CSRR-only) | 95.69 / (CSRR-only) | `measured` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

RML2018.01A remains a large fail under architecture freeze (long-sequence collapse). Hisar is CSRR-only measured. Default RML split is CSRR 50/10/40 vs TF AMR-Benchmark 6:2:2.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
