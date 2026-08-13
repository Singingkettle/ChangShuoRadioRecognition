# DAE — Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder

> S. Rajendran et al., "Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder", *IEEE Trans. Cogn. Commun. Netw. (2021)*.
> [https://ieeexplore.ieee.org/abstract/document/9487492](https://ieeexplore.ieee.org/abstract/document/9487492)

PyTorch / MMEngine port in CSRR. Algorithm short name **`dae`**
(= `configs/dae/` = `docs/dae/`).

## Method in one paragraph

LSTM auto-encoder with classification and reconstruction losses (CSRR `DAEHead`). Input is amplitude/phase with L2 on the amplitude channel.

## Paper section → code map

| paper | code |
|---|---|
| Network / backbone | `csrr/models/backbones/dae.py::DAE` |
| Train / test configs | `configs/dae/` |
| Shared AMC schedule / runtime | `configs/_base_/schedules/amc.py`, `configs/_base_/runtimes/amc.py` |
| Dataset loader | `csrr/datasets/amc.py::AMCDataset` |
| Input modality | A/P + reconstruction |

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
python tools/train.py configs/dae/dae_ap-deepsig-201610A.py

# test a checkpoint
python tools/test.py configs/dae/dae_ap-deepsig-201610A.py \
    work_dirs/<run>/best_accuracy_top1_epoch_*.pth
```

## Results

| Dataset | Overall (meas / target %) | Peak (meas / target %) | Status |
|---|---|---|---|
| RML2016.10A | 55.60 / 57.00 | 84.68 / 82.00 | `pass` |
| RML2016.10B | 63.20 / 62.00 | 93.24 / 85.00 | `pass` |
| RML2018.01A | 61.44 / 55.00 | 96.55 / 90.00 | `pass` |
| HisarMod | 54.27 / 40.00 | 61.39 / 70.00 | `fail` |

Numbers from the closed CSRR AMR campaign (approximate tolerances: overall ≥ target−2.0 pp, peak ≥ target−1.5 pp including near-match). Train entry points below are the official `configs/<algo>/` roots; some champions used local retunes that are **not** checked in.

## Documented deviations / notes

RML sets pass. Hisar peak fails (paper itself notes severe confusion on HisarMod). Overall can exceed the soft ~40% readout while peak stays short of ~70%.

Architecture freeze: retunes may change hyperparameters / init / schedule /
documented input pipeline only — not layer topology (except the documented
CLDNNW ZeroPad restoration).
