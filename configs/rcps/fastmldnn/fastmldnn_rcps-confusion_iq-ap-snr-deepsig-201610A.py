_base_ = ['../_base_/models/fastmldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/fastmldnn_rcps-confusion'
model = dict(head=dict(loss=dict(type='RCPSCrossEntropyLoss', reliability_key='snr', reliability_map=dict(type='linear', min=-20, max=18), epsilon=dict(type='power', max=1.0, gamma=1.0), base=dict(type='confusion', source='/home/citybuster/Data/RCPS/work_dirs/confusion_bases/deepsig201610A.npy', laplace=1e-4, temperature=1.0, prior_blend=0.5, prior=dict(type='uniform')), sample_weight=dict(type='none'), loss_weight=1.0)))
