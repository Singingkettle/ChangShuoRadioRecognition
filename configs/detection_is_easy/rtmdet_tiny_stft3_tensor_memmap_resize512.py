# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = './rtmdet_tiny_stft_3ch_resize512.py'

# Axis-B (model complexity): RTMDet-TINY on the hardshort-lowsnr STFT3 memmap, 57-class via --root coco_multiclass.
# Identical input/normalization to the M baseline; only the backbone scale differs (clean complexity ablation).
memmap_root = 'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap'

train_pipeline = [
    dict(type='LoadTensorMemmapFromCOCOStem', memmap_root=memmap_root, expected_channels=3),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadTensorMemmapFromCOCOStem', memmap_root=memmap_root, expected_channels=3),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'complex_stft_shape', 'tensor_memmap_shape', 'memmap_index',
        ),
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

work_dir = 'work_dirs/torchsig_mmdet_stft3_tensor_memmap_resize512_rtmdet_tiny'
