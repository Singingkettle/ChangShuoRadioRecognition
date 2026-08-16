# Reliability-power sample weighting route of the SNR-aware supervision spectrum (P2).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610B.py']

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='constant', value=0.0),
            base=dict(type='uniform'),
            sample_weight=dict(type='reliability_power', gamma=1.0, min=0.3),
            loss_weight=1.0)))
