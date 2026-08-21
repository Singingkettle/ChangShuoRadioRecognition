# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Paper: "Detection Is Easy, Recognition Is Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition", IEEE TCCN (under review).

_base_ = 'mmdet::gfl/gfl_r50_fpn_1x_coco.py'

# Axis-C cross-detector cell: GFL (R50-FPN, distribution-based one-stage) on the SAME STFT3 memmap input as the RTMDet cells,
# 57-class via --root coco_multiclass. Part of the detector-taxonomy sweep proving the
# predicted-box recipe and the detection-easy/recognition-hard pattern are family-independent.
# num_classes is left to the harness (recurses over the model dict). Ops that mmcv-lite lacks
# (RoIAlign / multi-scale deformable attention) run via the torchvision / pure-PyTorch fallbacks
# in run_mmdet_smoke.py, called by run_mmdet_train_eval.
custom_imports = dict(imports=['mmdet_plugins'], allow_failed_imports=False)
memmap_root = 'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap'

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='ComplexStftDetDataPreprocessor',
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],   # harness apply_tensor_stats overwrites from summary
        pad_size_divisor=32,
    ),
    backbone=dict(
        in_channels=3, frozen_stages=-1, norm_eval=False, init_cfg=None,
        style='pytorch',
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
)

_mm = ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
       'complex_stft_shape', 'tensor_memmap_shape', 'memmap_index')
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
    dict(type='PackDetInputs', meta_keys=_mm),
]

train_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=dict(classes=('signal',)),
                 filter_cfg=dict(filter_empty_gt=False, min_size=1), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=dict(classes=('signal',)), test_mode=True, pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=dict(classes=('signal',)), test_mode=True, pipeline=test_pipeline))

val_evaluator = dict(type='CocoMetric', metric='bbox')
test_evaluator = dict(type='CocoMetric', metric='bbox')
custom_hooks = []
load_from = None
resume = False
work_dir = 'work_dirs/torchsig_mmdet_stft3_memmap_gfl_stft3_memmap_resize512'
