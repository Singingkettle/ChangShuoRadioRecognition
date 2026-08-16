# RCPS hybrid soft-target route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['dscldnn_hard-ce_ap-iq-snr-hisar2019.py']

work_dir = 'work_dirs/amc/hisar2019/dscldnn_rcps-hybrid-eps01'
method_name = 'rcps_hybrid_eps01'
experiment_note = 'HisarMod2019.1 DSCLDNN retention-gated RCPS paired comparison after hard-CE baseline gate.'

model = dict(head=dict(loss=dict(
    type='RCPSCrossEntropyLoss',
    reliability_key='snr',
    reliability_map=dict(type='linear', min=-20, max=18),
    epsilon=dict(type='retention_power', max=0.1, gamma=2.0, retain_min=0.8),
    base=dict(type='uniform'),
    sample_weight=dict(type='none'),
    loss_weight=1.0)))
