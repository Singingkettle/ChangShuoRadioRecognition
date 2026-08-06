# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_m_stft_3ch_resize512.py'

# Axis-E (user-requested): complex-valued 1D CNN on raw IQ + in-graph FFT bridge (Design B).
# Raw IQ -> ComplexIQ1DPyramidBackbone (complex 1D stem/stages -> 3 complex taps -> per-tap reshape +
# FFT-along-time + [real,imag,abs] concat) -> 3 REAL maps [192,384,768] at 64/32/16 -> standard RTMDet-M
# neck+head. 57-class via --root coco_multiclass. Pass-through preprocessor hands raw IQ [B,2,N] to the
# backbone; the harness patch skips channel-forcing/stat-injection for RawIQPassThrough preprocessors.
raw_root = 'data/torchsig_hardshort_lowsnr_iq_65k_nvme/raw'

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='RawIQPassThroughDetDataPreprocessor',
        detector_input_shape=(512, 512),
    ),
    backbone=dict(
        _delete_=True,
        type='ComplexIQ1DPyramidBackbone',
        input_channels=2,
        complex_channels=(64, 128, 256),
        out_lengths=(4096, 1024, 256),
    ),
)

train_pipeline = [
    dict(type='LoadRawIQFromCOCOStem', raw_root=raw_root, target_shape=(512, 512)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadRawIQFromCOCOStem', raw_root=raw_root, target_shape=(512, 512)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'raw_iq_shape', 'raw_iq_path'),
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

work_dir = 'work_dirs/torchsig_mmdet_complexiq1d_fftbridge_rtmdet_m'
