_base_ = [
    '../_base_/models/petcgdnn_iq-snr-synthetic-awgn-v1.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/petcgdnn_critical-rcps'
method_name = 'critical_rcps_uniform'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(
                type='table',
                bins=[0.0, 0.20, 0.35, 0.65, 0.80, 1.0],
                values=[0.15, 0.35, 0.70, 0.70, 0.10, 0.0]),
            base=dict(type='uniform'),
            sample_weight=dict(type='none'),
            loss_weight=1.0),
    ),
)
