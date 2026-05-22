_base_ = [
    '../_base_/models/petcgdnn_iq-snr-synthetic-awgn-v1.py',
]

work_dir = '/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/petcgdnn_strict-awgn-dpc'
method_name = 'strict_awgn_dpc_rcps'

strict_dpc_source = '/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/teacher_posteriors/petcgdnn_strict_awgn_dpc_trainval.npz'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='retention_power', max=0.7, gamma=1.0, retain_min=0.8),
            base=dict(
                type='sample_posterior',
                source=strict_dpc_source,
                sample_index_key='global_sample_idx',
                laplace=1e-6,
                temperature=1.0,
                prior_blend=0.0),
            sample_weight=dict(type='none'),
            loss_weight=1.0),
    ),
)
