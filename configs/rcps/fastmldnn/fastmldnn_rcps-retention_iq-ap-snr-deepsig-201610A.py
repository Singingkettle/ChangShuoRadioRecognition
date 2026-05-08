_base_ = ['../_base_/models/fastmldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/fastmldnn_rcps-retention'
model = dict(head=dict(loss=dict(type='RCPSCrossEntropyLoss', reliability_key='snr', reliability_map=dict(type='linear', min=-20, max=18), epsilon=dict(type='retention_power', max=0.7, gamma=1.0, retain_min=0.8), base=dict(type='uniform'), sample_weight=dict(type='none'), loss_weight=1.0)))
