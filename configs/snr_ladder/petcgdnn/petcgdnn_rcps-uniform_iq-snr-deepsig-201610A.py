# RCPS uniform-base soft-target route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = [
    '../_base_/models/petcgdnn_iq-snr-deepsig-201610A.py',
]

work_dir = 'work_dirs/amc/deepsig201610A/petcgdnn_rcps-uniform'

method_name = 'rcps_uniform_power'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='power', max=1.0, gamma=1.0),
            base=dict(type='uniform'),
            sample_weight=dict(type='none'),
            loss_weight=1.0),
    ),
)
