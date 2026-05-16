_base_ = ['./cnn4_hard-ce_iq-snr-ucsd-rml22.py']

method_name = 'rcps_retention_eps0p1_g2'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=20),
            epsilon=dict(type='retention_power', max=0.1, gamma=2.0, retain_min=0.8),
            base=dict(type='uniform'),
            sample_weight=dict(type='none'),
            loss_weight=1.0,
        ),
    ),
)
