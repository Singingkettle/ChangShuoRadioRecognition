# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_stft_3ch_resize512.py'

# Online raw-IQ front end for the 2048-scene WBSig-clean-like split.  The COCO
# annotations can come from the matching tensor export because the loader maps
# train/tensors/train_000000.npy -> raw/train/train_000000.npz.
raw_root = 'data/torchsig_wbsig_clean_like_iq_2048/raw'

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='RawIQFilterbankDetDataPreprocessor',
        mean=[0.02671332743811264, 0.020614742426232315, 0.4077143975231623],
        std=[292.6175746738303, 288.14539273690355, 0.6617149383485579],
        pad_size_divisor=32,
        num_bins=512,
        kernel_size=513,
        stride=512,
        filterbank_init='fourier',
        filterbank_window='blackman-harris',
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

work_dir = 'work_dirs/iqdet_raw_iq_filterbank_rtmdet_tiny_2048_resize512_smoke'
