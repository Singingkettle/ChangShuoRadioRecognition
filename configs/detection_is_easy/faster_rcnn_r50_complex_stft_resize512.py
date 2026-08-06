# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Paper: "Detection Is Easy, Recognition Is Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition", IEEE TCCN (under review).

_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(imports=['mmdet_plugins'], allow_failed_imports=False)

data_root = 'data/torchsig_mini_complex_stft/coco/'
classes = ('signal',)
num_classes = 1

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='ComplexStftDetDataPreprocessor',
        mean=[0.0, 0.0],
        std=[1.0, 1.0],
        pad_size_divisor=32,
    ),
    backbone=dict(
        in_channels=2,
        frozen_stages=-1,
        norm_eval=False,
        init_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    roi_head=dict(bbox_head=dict(num_classes=num_classes)),
)

train_pipeline = [
    dict(type='LoadComplexStftFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadComplexStftFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'complex_stft_shape'),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/tensors/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/tensors/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/tensors/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox')
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_test.json', metric='bbox')

custom_hooks = []
load_from = None
resume = False
work_dir = 'work_dirs/torchsig_mmdet_complex_stft_resize512_faster_rcnn_r50'

