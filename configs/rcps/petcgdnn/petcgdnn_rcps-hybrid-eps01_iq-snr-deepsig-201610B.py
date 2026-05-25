_base_ = ['../_base_/models/petcgdnn_iq-snr-deepsig-201610B.py']
work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610B/petcgdnn_rcps-hybrid-eps01'
method_name = 'rcps_hybrid_eps01'
model = dict(head=dict(loss=dict(type='RCPSCrossEntropyLoss', reliability_key='snr', reliability_map=dict(type='linear', min=-20, max=18), epsilon=dict(type='retention_power', max=0.1, gamma=2.0, retain_min=0.8), base=dict(type='uniform'), sample_weight=dict(type='none'), loss_weight=1.0)))
