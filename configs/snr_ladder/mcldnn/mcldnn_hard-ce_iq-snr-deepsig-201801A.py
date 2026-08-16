# Hard cross-entropy baseline (the frozen model the ladder audits).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/mcldnn_iq-snr-deepsig-201801A.py']

work_dir = 'work_dirs/baseline_gate_2018A_mcldnn/amc/deepsig201801A/mcldnn_hard-ce'
method_name = 'hard_ce_baseline_gate_2018A_mcldnn'
