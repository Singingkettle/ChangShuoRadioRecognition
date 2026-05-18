# DPC-RCPS Implementation Status

Date: 2026-05-18

## Implemented

- Added sample-level posterior support to `RCPSCrossEntropyLoss` through `base.type='sample_posterior'`.
- `collect_predictions.py` now exports `sample_idx`, enabling sample-indexed teacher artifacts.
- Added `tools/rcps/build_sample_posterior.py` to convert prediction PKLs into `.npz` teacher posterior artifacts with `sample_idx`, `label`, `reliability`, `probs`, `logits`, `temperature`, and metadata.
- Added a PETCGDNN DPC-RCPS smoke config for RadioML2016.10A.

## Smoke Results

- Teacher posterior artifact:
  `/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201610A/petcgdnn_hard-ce_seed2026_train.npz`
- Smoke run:
  `/home/citybuster/Data/RCPS/work_dirs/dpc_smoke/amc/deepsig201610A/petcgdnn_dpc-rcps/seed_2026`
- Metrics:
  `/home/citybuster/Data/RCPS/work_dirs/dpc_smoke/metrics/deepsig201610A_petcgdnn_dpc-rcps_seed2026_test.csv`

The smoke run used one epoch only and is not paper evidence. It verifies that sample-posterior lookup, training, prediction export, and reliability-bin analysis complete end to end.

## Next

- Train DPC-RCPS with the same epoch budget and early stopping as the matched PETCGDNN hard-label baseline.
- Compare DPC-RCPS against RCPS-PosteriorBase on validation NLL/ECE/Brier and high-SNR retention before adding it to any main table.
