_base_ = ['./cgdnet_hard-ce_iq-snr-synthetic-awgn-v1.py']

work_dir = '/home/citybuster/Data/RCPS/work_dirs/synthetic_awgn/cgdnet_highsnr-teacher'
method_name = 'highsnr_teacher_hard_ce'

high_snr_filter = dict(type='FilterBySNR', save_range=(10, 18), preserve_classes=True)

train_dataloader = dict(dataset=dict(filter_cfg=high_snr_filter))
val_dataloader = dict(dataset=dict(filter_cfg=high_snr_filter))
test_dataloader = dict(dataset=dict(filter_cfg=high_snr_filter))
