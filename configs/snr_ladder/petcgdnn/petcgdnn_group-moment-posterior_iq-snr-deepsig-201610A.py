# Group-moment posterior route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610A.py']

work_dir = 'work_dirs/amc_group_rcps/deepsig201610A/petcgdnn_group-moment-posterior'
method_name = 'group_moment_rcps_posterior_base'

model = dict(
    head=dict(
        loss=dict(
            type='GroupMomentRCPSLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=18),
            epsilon=dict(type='retention_power', max=0.7, gamma=1.0, retain_min=0.8),
            base=dict(
                type='posterior_table',
                source='work_dirs/rcps_tables/deepsig201610A/petcgdnn_kerasinit_seed2026_reliability_base.npz',
                laplace=1e-4,
                temperature=1.0,
                prior_blend=0.5,
                prior=dict(type='uniform')),
            constraint_weight=0.5,
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
