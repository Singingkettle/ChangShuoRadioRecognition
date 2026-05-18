_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/dpc_smoke/amc/deepsig201610A/petcgdnn_dpc-entproj'
method_name = 'dpc_rcps_entropy_projected'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='retention_power', max=1.0, gamma=1.0, retain_min=0.8),
            base=dict(
                type='sample_posterior',
                source='/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201610A/petcgdnn_hard-ce_seed2026_train.npz',
                laplace=1e-4,
                temperature=1.5,
                prior_blend=1.0,
                prior=dict(type='uniform')),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
