_base_ = ['../_base_/models/mldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/mldnn_rcps-lowgate-c14-posterior-t2-b1p0-e0p5'

_posterior_base = '/home/citybuster/Data/RCPS/work_dirs/mldnn_supervision_400ep/posterior_bases/deepsig201610A_mldnn_hardce_validation_meanprob.npz'
_rcps_loss = dict(
    type='RCPSCrossEntropyLoss',
    reliability_key='snr_db',
    reliability_map=dict(type='linear', min=-20, max=18),
    epsilon=dict(type='low_reliability_power', max=0.5, gamma=1.0, cutoff=0.1578947368),
    base=dict(type="reliability_confusion", source=_posterior_base, laplace=1e-4, temperature=2.0, prior_blend=1.0, prior=dict(type="uniform")),
    sample_weight=dict(type='none'),
    loss_weight=1)
model = dict(head=dict(loss_amc_merge=_rcps_loss, loss_amc_ap=_rcps_loss, loss_amc_iq=_rcps_loss))
