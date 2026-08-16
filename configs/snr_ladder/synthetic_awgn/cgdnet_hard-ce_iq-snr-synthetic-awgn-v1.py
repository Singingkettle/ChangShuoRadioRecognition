# Hard cross-entropy baseline (the frozen model the ladder audits).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/cgdnet_iq-snr-synthetic-awgn-v1.py']

work_dir = 'work_dirs/synthetic_awgn/cgdnet_hard-ce'
method_name = 'hard_ce'

model = dict(head=dict(loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
