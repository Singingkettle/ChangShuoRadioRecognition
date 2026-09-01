# MLDNN — Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification

English | [简体中文](README_zh-CN.md)

> S. Chang et al., "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification," *IEEE Internet of Things Journal*, 2021. [IEEE 9462447](https://ieeexplore.ieee.org/document/9462447)

## Method in one paragraph

MLDNN jointly learns modulation recognition from I/Q and amplitude/phase views and an auxiliary high/low-SNR task. A learned SNR gate mixes the two modulation probabilities; the released loss evaluates that mixture in the log-probability domain.

## Paper section → code map

| paper | code |
|---|---|
| I/Q and A/P branches, SNR gate | `csrr/models/backbones/mldnn.py` |
| Four-task loss and prediction | `csrr/models/heads/mldnn_head.py` |
| I/Q to A/P definition | `csrr/datasets/transforms/processing.py` |
| 2016.10A protocol | `mldnn_iq-ap-deepsig-201610a.py`, `experiments/mldnn_iq-ap-deepsig-201610a_final.py` |
| 2018.01A protocol | `mldnn_iq-ap-deepsig-201801a.py`, `experiments/mldnn_iq-ap-deepsig-201801a_final.py` |
| Two-stage runner and checks | `reproduce.py`, `check_release.py` |

## Data

Download RadioML.2016.10A and RadioML.2018.01A from DeepSig, then convert them under `data/ModulationClassification/DeepSig/`. CSRR creates a stratified 50% train, 10% validation, and independent 40% test split; `train_and_validation.json` is the exact 60% union. The converter also builds packed I/Q caches, which avoid per-sample file reads.

```bash
python tools/convert_datasets/convert_amc.py \
  --data_root data/ModulationClassification
python configs/mldnn/check_release.py --check-data
```

## Train / evaluate

```bash
# 1. Install the measured environment and CSRR without dependency drift.
python -m pip install -r requirements/mldnn.txt
python -m pip install -e . --no-deps

# 2. Run validation selection, fresh 60% training, one test, and aggregation.
python configs/mldnn/reproduce.py --dataset all --devices 0 1 2

# 3. The shared entry points remain available for an inspected single run.
python tools/train.py configs/mldnn/mldnn_iq-ap-deepsig-201610a.py
python tools/train.py \
  configs/mldnn/experiments/mldnn_iq-ap-deepsig-201610a_final.py
python tools/test.py \
  configs/mldnn/experiments/mldnn_iq-ap-deepsig-201610a_final.py \
  work_dirs/<run>/epoch_<selected>.pth --phase-rotation-tta-views 8
```

The workflow selects the earliest epoch attaining maximum validation top-1, writes that choice atomically, trains a new model on the 60% union with validation disabled, and refuses a second test evaluation in the same run directory.

## Results

| Dataset | Published MAA | Reproduced MAA | Status |
|---|---:|---:|---|
| RadioML.2016.10A | 63.40% | 63.5841% | reproduced |
| RadioML.2018.01A | 60.70% | 60.7149% | reproduced |

Fixed aggregation rule: 2016.10A uses seeds 31/37/41, eight fixed phase views per model, and an equal probability mean; 2018.01A uses seed 17 and validation-selected epoch 370. No test-set weighting, member deletion, or result-dependent retry is used.

## Documented deviations / notes

Reproduction level: `statistical`.

The 50/10/40 two-stage protocol removes the test-as-validation behavior of the historical release. The measured 2016 path fixes mild optimization stabilization and a moving parameter average; both datasets use packed in-memory I/Q loading, strict MAA, and the paper's optimizer, learning rate, batch size, epoch bound, and four losses. Exact checkpoint bytes can vary with CUDA kernels, so acceptance is based on the predeclared aggregate MAA.

