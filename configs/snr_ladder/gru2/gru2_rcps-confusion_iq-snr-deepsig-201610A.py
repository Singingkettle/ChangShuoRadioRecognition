# RCPS confusion-base soft-target route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/gru2_iq-snr-deepsig-201610A.py']

work_dir = 'work_dirs/amc/deepsig201610A/gru2_rcps-confusion'
method_name = 'rcps_confusion_power'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='retention_power', max=0.7, gamma=1.0, retain_min=0.8),
            base=dict(
                type='confusion',
                source='work_dirs/confusion_bases/deepsig201610A_gru2_seed2026.npy',
                laplace=1e-4,
                prior_blend=0.5,
                prior=dict(type='uniform')),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
