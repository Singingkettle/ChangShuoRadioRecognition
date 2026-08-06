# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_m_stft_3ch_resize512.py'

raw_root = 'data/torchsig_latest_clean_placeholder/raw'



model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='RawIQFilterbankDetDataPreprocessor',
        mean=[-0.010288547558199745, -0.018919106865205926, 0.18054384734568885],
        std=[218.39321626626995, 216.01599153448888, 0.339832409195686],
        pad_size_divisor=32,
        num_bins=512,
        kernel_size=513,
        stride=512,
        filterbank_init='fourier',
        filterbank_window='hann',
        channel_mode='realimag_logmag',
        trainable_filterbank=True,
        output_size=(512, 512),
    ),
    backbone=dict(input_channels=3),
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
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'raw_iq_shape',
            'raw_iq_path',
        ),
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))


optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys={
            'data_preprocessor.filterbank': dict(lr_mult=1.0, decay_mult=0.0),
        },
    )
)

work_dir = 'work_dirs/profile_raw_m'
