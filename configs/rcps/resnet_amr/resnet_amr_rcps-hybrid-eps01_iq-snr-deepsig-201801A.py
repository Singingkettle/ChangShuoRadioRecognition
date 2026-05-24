_base_ = ['../_base_/models/resnet_amr_iq-snr-deepsig-201801A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/rcps_main_2018A/amc/deepsig201801A/resnet_amr_rcps-hybrid-eps01'
method_name = 'rcps_hybrid_eps01_reliability_confusion'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=30),
            epsilon=dict(type='retention_power', max=0.1, gamma=2.0, retain_min=0.8),
            base=dict(
                type='reliability_confusion',
                source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201801A/resnet_amr_hard-ce_seed2026_reliability_base.npz',
                laplace=1e-4,
                temperature=1.0,
                prior_blend=0.5,
                prior=dict(type='uniform')),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))

experiment_note = 'Conservative RCPS-Hybrid candidate for ResNet-AMR 2018A; activate only after hard-CE gate and teacher base generation.'
