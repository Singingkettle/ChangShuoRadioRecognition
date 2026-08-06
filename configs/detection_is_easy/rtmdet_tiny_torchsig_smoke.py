# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

data_root = 'data/torchsig_mini/coco/'
classes = ('signal',)
num_classes = 1

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=num_classes),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
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
        data_prefix=dict(img='train/images/'),
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
        data_prefix=dict(img='val/images/'),
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
        data_prefix=dict(img='test/images/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
)

load_from = None
resume = False
work_dir = 'work_dirs/torchsig_mmdet_smoke'
