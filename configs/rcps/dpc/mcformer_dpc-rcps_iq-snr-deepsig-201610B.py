_base_ = ['../_base_/models/mcformer_iq-snr-deepsig-201610B.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/dpc_smoke/amc/deepsig201610B/mcformer_dpc-rcps'
method_name = 'dpc_rcps_sample_posterior'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='retention_power', max=0.7, gamma=1.0, retain_min=0.8),
            base=dict(
                type='sample_posterior',
                source='/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201610B/mcformer_hard-ce_seed2026_train.npz',
                laplace=1e-4,
                temperature=1.0,
                prior_blend=0.25,
                prior=dict(type='uniform')),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
