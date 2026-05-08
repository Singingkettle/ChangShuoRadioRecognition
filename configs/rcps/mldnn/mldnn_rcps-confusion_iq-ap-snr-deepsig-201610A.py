_base_ = ['../_base_/models/mldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/mldnn_rcps-confusion'

_rcps_loss = dict(type='RCPSCrossEntropyLoss', reliability_key='snr_db', reliability_map=dict(type='linear', min=-20, max=18), epsilon=dict(type='power', max=1.0, gamma=1.0), base=dict(type='confusion', source='/home/citybuster/Data/RCPS/work_dirs/confusion_bases/deepsig201610A.npy', laplace=1e-4, temperature=1.0, prior_blend=0.5, prior=dict(type='uniform')), sample_weight=dict(type='none'), loss_weight=1)
model = dict(head=dict(loss_amc_merge=_rcps_loss, loss_amc_ap=_rcps_loss, loss_amc_iq=_rcps_loss))
