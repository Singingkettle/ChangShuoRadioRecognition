# High-SNR teacher distillation route on the synthetic AWGN anchor benchmark.
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = ['./petcgdnn_hard-ce_iq-snr-synthetic-awgn-v1.py']

work_dir = 'work_dirs/synthetic_awgn/petcgdnn_highsnr-teacher'
method_name = 'highsnr_teacher_hard_ce'

high_snr_filter = dict(type='FilterBySNR', save_range=(10, 18), preserve_classes=True)

train_dataloader = dict(dataset=dict(filter_cfg=high_snr_filter))
val_dataloader = dict(dataset=dict(filter_cfg=high_snr_filter))
test_dataloader = dict(dataset=dict(filter_cfg=high_snr_filter))
