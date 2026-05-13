_base_ = ['./mldnn_rcps-lowgate-c14_iq-ap-snr-deepsig-201610A.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/strict_split_400ep/amc/deepsig201610A/mldnn_rcps-lowgate-c14-strict'

train_dataloader = dict(dataset=dict(ann_file='train.json'))
val_dataloader = dict(dataset=dict(ann_file='validation.json'))
test_dataloader = dict(dataset=dict(ann_file='test.json'))
