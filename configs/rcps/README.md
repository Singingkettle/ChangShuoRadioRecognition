# RCPS Experiments

This folder contains the AMC implementation for Reliability-Conditioned
Posterior Supervision (RCPS).

## Main AMC Matrix

The RCPS configs use the server data root directly:
`/home/citybuster/Data/WirelessRadio/data/ModulationClassification`.

Generate the commands for the TPAMI main table:

```bash
python tools/rcps/run_amc_matrix.py
```

Execute the same matrix:

```bash
python tools/rcps/run_amc_matrix.py --execute --test
```

For a quick smoke test:

```bash
python tools/rcps/run_amc_matrix.py \
  --models cnn2 \
  --methods hard-ce rcps-uniform \
  --seeds 2026 \
  --max-epochs 1 \
  --execute --test
```

## Prior and Confusion Bases

Build the prior base from the training annotation:

```bash
python tools/rcps/build_prior.py \
  /home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10A/train.json \
  --out /home/citybuster/Data/RCPS/work_dirs/priors/deepsig201610A.npy
```

Build the confusion-aware base from a hard-label validation run:

```bash
python tools/rcps/build_confusion_base.py \
  /home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/cnn2_hard-ce/seed_2026/res/paper.pkl \
  --out /home/citybuster/Data/RCPS/work_dirs/confusion_bases/deepsig201610A.npy
```

## Reliability Metrics

Summarize reliability-stratified metrics from `paper.pkl` files:

```bash
python tools/rcps/analyze_reliability.py \
  --run deepsig201610A=cnn2=hard-ce=work_dirs/rcps/amc/deepsig201610A/cnn2_hard-ce/seed_2026/res/paper.pkl \
  --run deepsig201610A=cnn2=rcps-uniform=work_dirs/rcps/amc/deepsig201610A/cnn2_rcps-uniform/seed_2026/res/paper.pkl \
  --out-csv work_dirs/rcps/metrics/deepsig201610A_cnn2_seed2026.csv
```

The resulting CSV contains accuracy, NLL, ECE, Brier score, mean confidence,
and entropy for every SNR bin.

## Data Preparation

Write the AMC data manifest:

```bash
python tools/rcps/prepare_amc_manifest.py
```

Prepare CIFAR-10-C annotations:

```bash
python tools/rcps/prepare_cifar10c.py --download
```

Prepare Speech Commands v0.02 reliability annotations:

```bash
python tools/rcps/prepare_speech_commands.py --download
```

## Cross-Modal Protocol

The shared RCPS target builder is task independent. For the paper's
cross-modal stage, reuse the same target rule in a CIFAR-10 corruption runner
and a Speech Commands noisy-audio runner. In those runners, pass corruption
severity or synthetic SNR as the reliability coordinate, then export the same
CSV schema produced by `tools/rcps/analyze_reliability.py`.
