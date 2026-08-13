# CNN1DPF — Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks

> S. Peng et al. / AMR-Benchmark 1DCNN-PF, "Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks", *ISSCS / related (AMR-Benchmark port)*.
> [https://lirias.kuleuven.be/retrieve/546033](https://lirias.kuleuven.be/retrieve/546033)

PyTorch / MMEngine port in CSRR. Algorithm short name **`cnn1dpf`**
(= `configs/cnn1dpf/` = `docs/cnn1dpf/`).

## Method in one paragraph

Parallel-fusion 1-D CNN: amplitude and phase streams (CSRR feeds A/P branches to match TF `to_amp_phase`) are convolved separately and fused for classification.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/cnn1dpf.py::CNN1DPF` |
| Train / test configs | `configs/cnn1dpf/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P (parallel branches) |

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
python tools/train.py configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/cnn1dpf/cnn1dpf_iq-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 57.56 / 57.00 | 90.52 / 85.00 | `pass` |
| RML2016.10B | 58.45 / 62.00 | 89.62 / 88.00 | `fail` |
| RML2018.01A | 55.95 / 57.00 | 90.87 / 91.00 | `pass` |
| HisarMod | 42.18 / (CSRR-only) | 65.97 / (CSRR-only) | `measured` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

10B overall still short of the approximate bar under 50/10/40. Hisar is CSRR-only measured. TF also uses A/P — do not switch to raw I/Q.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
