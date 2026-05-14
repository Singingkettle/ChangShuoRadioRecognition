_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/petcgdnn_rcps-posterior'
method_name = 'rcps_posterior_base'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='retention_power', max=0.7, gamma=1.0, retain_min=0.8),
            base=dict(
                type='posterior_table',
                source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610A/petcgdnn_hard-ce_seed2026_reliability_base.npz',
                laplace=1e-4,
                temperature=1.0,
                prior_blend=0.5,
                prior=dict(type='uniform')),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
