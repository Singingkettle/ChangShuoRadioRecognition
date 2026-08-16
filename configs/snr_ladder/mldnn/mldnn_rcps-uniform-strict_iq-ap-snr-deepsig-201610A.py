# RCPS uniform-base soft-target route of the audited spectrum.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['./mldnn_rcps-uniform_iq-ap-snr-deepsig-201610A.py']

work_dir = 'work_dirs/strict_split_400ep/amc/deepsig201610A/mldnn_rcps-uniform-strict'

train_dataloader = dict(dataset=dict(ann_file='train.json'))
val_dataloader = dict(dataset=dict(ann_file='validation.json'))
test_dataloader = dict(dataset=dict(ann_file='test.json'))
