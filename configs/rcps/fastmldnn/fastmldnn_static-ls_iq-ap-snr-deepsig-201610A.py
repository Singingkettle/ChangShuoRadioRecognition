_base_ = ['../_base_/models/fastmldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/fastmldnn_static-ls'
model = dict(head=dict(loss=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1.0)))
