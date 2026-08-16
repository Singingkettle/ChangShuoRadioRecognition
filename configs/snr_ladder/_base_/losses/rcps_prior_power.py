method_name = 'rcps_prior_power'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='power', max=1.0, gamma=1.0),
            base=dict(
                type='prior',
                source='work_dirs/rcps/priors/deepsig201610A.npy'),
            sample_weight=dict(type='none'),
            loss_weight=1.0),
    ),
)
