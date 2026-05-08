_base_ = ['../_base_/models/mldnn_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/mldnn_static-ls'

model = dict(head=dict(loss_amc_merge=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1), loss_amc_ap=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1), loss_amc_iq=dict(type='LabelSmoothingCrossEntropyLoss', smoothing=0.1, loss_weight=1)))
