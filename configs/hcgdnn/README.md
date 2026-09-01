# HCGDNN — A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification

English | [简体中文](README_zh-CN.md)

> S. Chang et al., "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification," *IEEE Transactions on Wireless Communications*, 2022. [IEEE 9764618](https://ieeexplore.ieee.org/document/9764618)

## Method in one paragraph

HCGDNN forms a hierarchy from a convolutional representation and two stacked bidirectional GRU representations. Three classification heads are trained jointly, while validation predictions determine a constrained non-negative fusion whose weights sum to one.

## Paper section → code map

| paper | code |
|---|---|
| CNN and hierarchical GRU backbone | `csrr/models/backbones/hcgdnn.py` |
| Three heads and probability fusion | `csrr/models/heads/hcgdnn_head.py` |
| Fusion objective and constrained solver | `csrr/evaluation/metrics/hcgdnn.py` |
| Validation-selection config | `hcgdnn_iq-deepsig-201610a.py` |
| Fresh 60% final config | `experiments/hcgdnn_iq-deepsig-201610a_final.py` |
| Checkpoint processing and runner | `release_utils.py`, `reproduce.py` |

## Data

Download RadioML.2016.10A from DeepSig and convert it under `data/ModulationClassification/DeepSig/`. CSRR creates a modulation-SNR-stratified 50% train, 10% validation, and independent 40% test split. `train_and_validation.json` is the exact 60% union. With `cache=True`, each process loads its full split into host memory before training.

```bash
python tools/convert_datasets/convert_amc.py \
  --data_root data/ModulationClassification
python configs/hcgdnn/check_release.py --check-data
```

## Train / evaluate

```bash
# 1. Install the measured environment and CSRR without dependency drift.
python -m pip install -r requirements/hcgdnn.txt
python -m pip install -e . --no-deps

# 2. Run validation selection, fresh 60% training, one test, and aggregation.
python configs/hcgdnn/reproduce.py --devices 0 1 2

# 3. The shared entry points remain available for an inspected single run.
python tools/train.py configs/hcgdnn/hcgdnn_iq-deepsig-201610a.py
python tools/train.py \
  configs/hcgdnn/experiments/hcgdnn_iq-deepsig-201610a_final.py
python tools/test.py \
  configs/hcgdnn/experiments/hcgdnn_iq-deepsig-201610a_final.py \
  work_dirs/<run>/averaged_calibrated.pth
```

The workflow atomically records the earliest epoch attaining maximum validation top-1 and its validation-derived fusion. It then trains from scratch on the 60% union with validation disabled, averages the last three retained checkpoints, transplants the frozen fusion, and refuses a second test in the same run directory.

## Results

| Dataset | Published MAA | Reproduced MAA | Status |
|---|---:|---:|---|
| RadioML.2016.10A | 63.75% | 63.7864% | reproduced |

Fixed aggregation rule: seeds 31/37/41/43/47/53 each use the equal parameter mean of their last three retained final checkpoints; the six resulting prediction tensors are then averaged with equal probability weights. No test-set weighting, member deletion, or result-dependent retry is used.

## Documented deviations / notes

Reproduction level: `statistical`.

The 50/10/40 two-stage protocol removes the test-as-validation behavior of the historical release. The measured path fixes mild optimization stabilization and checkpoint averaging while retaining the paper's optimizer, learning rate, batch size, 1600-epoch bound, three losses, full-sample fusion objective, and constrained solver. Exact checkpoint bytes can vary with CUDA kernels, so acceptance is based on the predeclared six-seed aggregate MAA.

