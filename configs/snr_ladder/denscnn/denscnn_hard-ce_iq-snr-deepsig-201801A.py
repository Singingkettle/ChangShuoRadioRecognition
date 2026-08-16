# Hard cross-entropy baseline (the frozen model the ladder audits).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/denscnn_iq-snr-deepsig-201801A.py']

work_dir = 'work_dirs/baseline_gate_2018A/amc/deepsig201801A/denscnn_hard-ce'
method_name = 'hard_ce'

experiment_note = 'Fallback baseline gate for AMR-Benchmark style DenseNet-AMR/DensCNN on RadioML2018.01A with SNR metadata.'
