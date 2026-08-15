# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Paper: "Detection Is Easy, Recognition Is Hard: Rethinking Vision-Based Wideband
# Signal Detection and Recognition" IEEE TCCN (under review).
# YOLOX-S on the hardshort-lowsnr STFT3 memmap (57-class via --root coco_multiclass).
# A third detector family for the cross-detector generalization of the predicted-box
# recipe. Reads the packed memmap like the RTMDet configs, not per-image .npy tensors.
_base_ = './yolox_s_stft_3ch_resize512_short20.py'

memmap_root = 'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap'
img_scale = (512, 512)

train_pipeline = [
    dict(type='LoadTensorMemmapFromCOCOStem', memmap_root=memmap_root, expected_channels=3),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadTensorMemmapFromCOCOStem', memmap_root=memmap_root, expected_channels=3),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
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
