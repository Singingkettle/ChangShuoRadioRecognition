# RCPS Experiments

This folder contains the AMC implementation for Reliability-Conditioned
Posterior Supervision (RCPS).


## Baseline-First Gate

Current TPAMI experiments must start from hard-label baseline parity. The
previous 10-epoch `main_10ep_3seed` run is diagnostic-invalid and must not be
used as paper evidence. Use the reference registry before launching RCPS runs:

```bash
python tools/rcps/audit_framework_parity.py \
  configs/mldnn/mldnn_iq-ap-deepsig201610A.py \
  configs/rcps/mldnn/mldnn_hard-ce_iq-ap-snr-deepsig-201610A.py

python tools/rcps/check_baseline_gate.py
```

The first hard gate is `MLDNN + RadioML2016.10A`: reproduce about 0.63 overall
accuracy under the original 400-epoch schedule before comparing RCPS variants.
`RCPS-Retention` is available for high-reliability preservation experiments.

## Main AMC Matrix

The RCPS configs use the server data root directly:
`/home/citybuster/Data/WirelessRadio/data/ModulationClassification`.

Generate the commands for the TPAMI main table:

```bash
python tools/rcps/run_amc_matrix.py
```

The default matrix now uses the stronger maintained baselines
`MLDNN`, `FastMLDNN`, `CGDNet`, `PETCGDNN`, and `MCformer`. `MCLDNN` and `CNN2` are kept as diagnostic or ablation models only.

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

## Calibration Grid

Run a one-seed calibration pilot on the stronger AMC models:

```bash
python tools/rcps/run_calibration_grid.py \
  --models mldnn fastmldnn cgdnet petcgdnn mcformer \
  --methods hard-ce rcps-uniform \
  --seeds 2026 \
  --max-epochs 1 \
  --epsilon-max 0.3 0.5 0.7 1.0 \
  --epsilon-gamma 0.5 1.0 2.0 \
  --collect-splits validation \
  --analyze \
  --execute
```

For a smaller pilot, pass one candidate such as `--epsilon-max 0.5
--epsilon-gamma 2.0`. Select candidates with:

```bash
python tools/rcps/select_calibration.py \
  --metrics /home/citybuster/Data/RCPS/work_dirs/metrics/calibration/*.csv \
  --out-csv /home/citybuster/Data/RCPS/work_dirs/metrics/calibration/selected.csv
```

## Monitoring

Write one status snapshot:

```bash
python tools/rcps/monitor_experiments.py
```

Run repeated snapshots every ten minutes:

```bash
nohup bash tools/rcps/monitor_loop.sh \
  > /home/citybuster/Data/RCPS/work_dirs/logs/rcps_monitor_loop.nohup.log 2>&1 &
```

Snapshots are appended to
`/home/citybuster/Data/RCPS/work_dirs/logs/rcps_monitor_snapshots.jsonl`
and
`/home/citybuster/Data/RCPS/work_dirs/logs/rcps_monitor_snapshots.md`.

## Selected Main Runs

Build the per-family selected config table from calibration metrics:

```bash
python tools/rcps/select_main_configs.py
```

Run selected 3-seed main experiments:

```bash
python tools/rcps/run_selected_main.py \
  --seeds 2026 2027 2028 \
  --max-epochs 10 \
  --collect-splits validation test \
  --analyze \
  --skip-existing \
  --execute
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
