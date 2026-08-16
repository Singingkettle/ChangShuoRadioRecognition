# RCPS hybrid soft-target route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/petcgdnn_iq-snr-hisar2019.py']
work_dir = 'work_dirs/amc/hisar2019/petcgdnn_rcps-hybrid-eps01'
method_name = 'rcps_hybrid_eps01'
model = dict(head=dict(loss=dict(type='RCPSCrossEntropyLoss', reliability_key='snr', reliability_map=dict(type='linear', min=-20, max=18), epsilon=dict(type='retention_power', max=0.1, gamma=2.0, retain_min=0.8), base=dict(type='uniform'), sample_weight=dict(type='none'), loss_weight=1.0)))
