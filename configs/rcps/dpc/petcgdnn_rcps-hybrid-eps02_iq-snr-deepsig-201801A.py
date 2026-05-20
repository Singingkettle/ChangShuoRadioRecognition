_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201801A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/rcps_hybrid_2018A/amc/deepsig201801A/petcgdnn_rcps-hybrid-eps02'
method_name = 'rcps_hybrid_eps02_reliability_confusion'

model = dict(
    head=dict(
        loss=dict(
            type='RCPSCrossEntropyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=30),
            epsilon=dict(type='retention_power', max=0.2, gamma=2.0, retain_min=0.8),
            base=dict(
                type='reliability_confusion',
                source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201801A/petcgdnn_hard-ce_seed2026_reliability_base.npz',
                laplace=1e-4,
                temperature=1.0,
                prior_blend=0.5,
                prior=dict(type='uniform')),
            sample_weight=dict(type='none'),
            loss_weight=1.0)))

experiment_note = 'Slightly stronger class-level RCPS-Hybrid pilot for 2018A with high-reliability retention.'
