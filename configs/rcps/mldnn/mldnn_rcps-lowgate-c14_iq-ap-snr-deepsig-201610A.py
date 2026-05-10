_base_ = ['../_base_/models/mldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/mldnn_rcps-lowgate-c14'

_rcps_loss = dict(
    type='RCPSCrossEntropyLoss',
    reliability_key='snr_db',
    reliability_map=dict(type='linear', min=-20, max=18),
    epsilon=dict(type='low_reliability_power', max=0.7, gamma=1.0, cutoff=0.1578947368),
    base=dict(type='uniform'),
    sample_weight=dict(type='none'),
    loss_weight=1)
model = dict(head=dict(loss_amc_merge=_rcps_loss, loss_amc_ap=_rcps_loss, loss_amc_iq=_rcps_loss))
