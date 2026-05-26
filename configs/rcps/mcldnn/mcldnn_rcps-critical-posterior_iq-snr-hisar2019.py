_base_ = ['../_base_/models/mcldnn_iq-snr-hisar2019.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/hisar2019/mcldnn_rcps-critical-posterior'
method_name = 'rcps_critical_posterior'

model = dict(head=dict(loss=dict(
    type='RCPSCrossEntropyLoss',
    reliability_key='snr',
    reliability_map=dict(type='linear', min=-20, max=18),
    epsilon=dict(
        type='table',
        bins=[0.0, 0.25, 0.37, 0.50, 0.65, 0.80, 1.0],
        values=[0.0, 0.0, 0.025, 0.05, 0.04, 0.015, 0.0]),
    base=dict(
        type='posterior_table',
        source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/hisar2019/mcldnn_hardce_3seed_validation_meanprob_t2.npz',
        laplace=1e-4,
        temperature=1.0,
        prior_blend=0.25,
        prior=dict(type='uniform')),
    sample_weight=dict(type='none'),
    loss_weight=1.0)))
