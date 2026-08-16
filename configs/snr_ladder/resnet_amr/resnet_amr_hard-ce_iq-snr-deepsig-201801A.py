# Hard cross-entropy baseline (the frozen model the ladder audits).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/resnet_amr_iq-snr-deepsig-201801A.py']

work_dir = 'work_dirs/baseline_gate_2018A/amc/deepsig201801A/resnet_amr_hard-ce'
method_name = 'hard_ce'

experiment_note = 'Baseline gate for AMR-Benchmark style ResNet-AMR on RadioML2018.01A with SNR metadata.'
