_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610B.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610B/petcgdnn_rcps-critical-posterior'
method_name = 'rcps_critical_posterior'

model = dict(head=dict(loss=dict(
    type='RCPSCrossEntropyLoss',
    reliability_key='snr',
    reliability_map=dict(type='linear', min=-20, max=18),
    # Critical-band gate: off at the low-SNR floor and high-SNR plateau,
    # light posterior correction around the empirical 10B waterfall band.
    epsilon=dict(
        type='table',
        bins=[0.0, 0.21, 0.32, 0.42, 0.53, 0.68, 1.0],
        values=[0.0, 0.0, 0.05, 0.08, 0.05, 0.0, 0.0]),
    base=dict(
        type='posterior_table',
        source='/home/citybuster/Data/RCPS/work_dirs/rcps_tables/deepsig201610B/petcgdnn_hardce_3seed_validation_meanprob_t2.npz',
        laplace=1e-4,
        temperature=1.0,
        prior_blend=0.25,
        prior=dict(type='uniform')),
    sample_weight=dict(type='none'),
    loss_weight=1.0)))
