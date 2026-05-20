_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201801A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/dpc_consistency/amc/deepsig201801A/petcgdnn_dpc-consistency-eps03'
method_name = 'dpc_posterior_consistency_eps03'

model = dict(
    head=dict(
        loss=dict(
            type='DPCConsistencyLoss',
            reliability_key='snr',
            reliability_map=dict(type='linear', min=-20, max=30),
            # Hard CE remains the primary objective. The posterior-path teacher
            # acts only as a low-reliability consistency regularizer.
            epsilon=dict(type='low_reliability_power', max=0.3, gamma=2.0, cutoff=0.4),
            base=dict(
                type='sample_posterior',
                source='/home/citybuster/Data/RCPS/work_dirs/dpc_teacher_posteriors/deepsig201801A/petcgdnn_hard-ce_seed2026_train.npz',
                laplace=1e-5,
                temperature=0.75,
                prior_blend=0.10,
                prior=dict(type='uniform')),
            consistency_weight=0.1,
            sample_weight=dict(type='none'),
            loss_weight=1.0)))
