# RCPS critical-band posterior route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/petcgdnn_iq-snr-hisar2019.py']

work_dir = 'work_dirs/amc/hisar2019/petcgdnn_rcps-critical-posterior'
method_name = 'rcps_critical_posterior'

model = dict(head=dict(loss=dict(
    type='RCPSCrossEntropyLoss',
    reliability_key='snr',
    reliability_map=dict(type='linear', min=-20, max=18),
    # Hisar has a broad reliability transition; use a smaller gate than 10B.
    epsilon=dict(
        type='table',
        bins=[0.0, 0.25, 0.37, 0.50, 0.65, 0.80, 1.0],
        values=[0.0, 0.0, 0.025, 0.05, 0.04, 0.015, 0.0]),
    base=dict(
        type='posterior_table',
        source='work_dirs/rcps_tables/hisar2019/petcgdnn_hardce_3seed_validation_meanprob_t2.npz',
        laplace=1e-4,
        temperature=1.0,
        prior_blend=0.25,
        prior=dict(type='uniform')),
    sample_weight=dict(type='none'),
    loss_weight=1.0)))
