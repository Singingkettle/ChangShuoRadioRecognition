# Copyright (c) Shuo Chang and contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Paper: "Detection Is Easy, Recognition Is Hard: Rethinking Vision-Based Wideband Signal Detection and Recognition", IEEE Transactions on Wireless Communications (in preparation).

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.004
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
classes = (
    '1024qam',
    '128qam_cross',
    '16ask',
    '16fsk',
    '16gfsk',
    '16gmsk',
    '16msk',
    '16psk',
    '16qam',
    '2fsk',
    '2gfsk',
    '2gmsk',
    '2msk',
    '256qam',
    '32ask',
    '32psk',
    '32qam',
    '32qam_cross',
    '4ask',
    '4fsk',
    '4gfsk',
    '4gmsk',
    '4msk',
    '512qam_cross',
    '64ask',
    '64psk',
    '64qam',
    '8ask',
    '8fsk',
    '8gfsk',
    '8gmsk',
    '8msk',
    '8psk',
    'am-dsb',
    'am-dsb-sc',
    'am-lsb',
    'am-usb',
    'bpsk',
    'chirpss',
    'fm',
    'lfm-data',
    'lfm-radar',
    'ofdm-1024',
    'ofdm-1200',
    'ofdm-128',
    'ofdm-180',
    'ofdm-2048',
    'ofdm-256',
    'ofdm-300',
    'ofdm-512',
    'ofdm-600',
    'ofdm-64',
    'ofdm-72',
    'ofdm-900',
    'ook',
    'qpsk',
    'tone',
)
custom_hooks = []
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet_plugins',
    ])
data_root = 'data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet', interval=5, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=20, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
device = 'cuda:0'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
memmap_root = 'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap'
model = dict(
    _scope_='mmdet',
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=1.0,
        expand_ratio=0.5,
        init_cfg=None,
        input_channels=3,
        norm_cfg=dict(type='SyncBN'),
        type='ComplexStftCSPNeXt',
        widen_factor=1.0),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=False,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=57,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        mean=[
            2.4033697286739977e-13,
            -1.4910969465406423e-12,
            1.211290168174614,
        ],
        pad_size_divisor=32,
        std=[
            12.82534791330875,
            12.830129644721533,
            0.765134411640441,
        ],
        type='ComplexStftDetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=2,
        out_channels=256,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=1000,
        score_thr=0.01),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
num_classes = 57
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=False,
        end=1000,
        start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=150,
        _scope_='mmdet',
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=20262811)
resume = False
stage2_num_epochs = 20
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_test.json',
        backend_args=None,
        data_prefix=dict(img='test/tensors/'),
        data_root=
        'data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass/',
        metainfo=dict(
            classes=(
                '1024qam',
                '128qam_cross',
                '16ask',
                '16fsk',
                '16gfsk',
                '16gmsk',
                '16msk',
                '16psk',
                '16qam',
                '2fsk',
                '2gfsk',
                '2gmsk',
                '2msk',
                '256qam',
                '32ask',
                '32psk',
                '32qam',
                '32qam_cross',
                '4ask',
                '4fsk',
                '4gfsk',
                '4gmsk',
                '4msk',
                '512qam_cross',
                '64ask',
                '64psk',
                '64qam',
                '8ask',
                '8fsk',
                '8gfsk',
                '8gmsk',
                '8msk',
                '8psk',
                'am-dsb',
                'am-dsb-sc',
                'am-lsb',
                'am-usb',
                'bpsk',
                'chirpss',
                'fm',
                'lfm-data',
                'lfm-radar',
                'ofdm-1024',
                'ofdm-1200',
                'ofdm-128',
                'ofdm-180',
                'ofdm-2048',
                'ofdm-256',
                'ofdm-300',
                'ofdm-512',
                'ofdm-600',
                'ofdm-64',
                'ofdm-72',
                'ofdm-900',
                'ook',
                'qpsk',
                'tone',
            )),
        pipeline=[
            dict(
                expected_channels=3,
                memmap_root=
                'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap',
                type='LoadTensorMemmapFromCOCOStem'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'complex_stft_shape',
                    'tensor_memmap_shape',
                    'memmap_index',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file=
    'data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass/annotations/instances_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    outfile_prefix=
    'work_dirs/baseline_mc_rtmdet_m_20e_seed20262811/source_data/test_predictions',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(
        expected_channels=3,
        memmap_root='data/torchsig_hardshort_lowsnr_stft3_memmap/memmap',
        type='LoadTensorMemmapFromCOCOStem'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'complex_stft_shape',
            'tensor_memmap_shape',
            'memmap_index',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet',
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=20,
    type='EpochBasedTrainLoop',
    val_interval=5)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=8,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/tensors/'),
        data_root=
        'data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass/',
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        metainfo=dict(
            classes=(
                '1024qam',
                '128qam_cross',
                '16ask',
                '16fsk',
                '16gfsk',
                '16gmsk',
                '16msk',
                '16psk',
                '16qam',
                '2fsk',
                '2gfsk',
                '2gmsk',
                '2msk',
                '256qam',
                '32ask',
                '32psk',
                '32qam',
                '32qam_cross',
                '4ask',
                '4fsk',
                '4gfsk',
                '4gmsk',
                '4msk',
                '512qam_cross',
                '64ask',
                '64psk',
                '64qam',
                '8ask',
                '8fsk',
                '8gfsk',
                '8gmsk',
                '8msk',
                '8psk',
                'am-dsb',
                'am-dsb-sc',
                'am-lsb',
                'am-usb',
                'bpsk',
                'chirpss',
                'fm',
                'lfm-data',
                'lfm-radar',
                'ofdm-1024',
                'ofdm-1200',
                'ofdm-128',
                'ofdm-180',
                'ofdm-2048',
                'ofdm-256',
                'ofdm-300',
                'ofdm-512',
                'ofdm-600',
                'ofdm-64',
                'ofdm-72',
                'ofdm-900',
                'ook',
                'qpsk',
                'tone',
            )),
        pipeline=[
            dict(
                expected_channels=3,
                memmap_root=
                'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap',
                type='LoadTensorMemmapFromCOCOStem'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        expected_channels=3,
        memmap_root='data/torchsig_hardshort_lowsnr_stft3_memmap/memmap',
        type='LoadTensorMemmapFromCOCOStem'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmdet',
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(_scope_='mmdet', crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(_scope_='mmdet', type='YOLOXHSVRandomAug'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(
        _scope_='mmdet',
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='Pad'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
tta_model = dict(
    _scope_='mmdet',
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmdet',
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        960,
                        960,
                    ),
                    type='Pad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='val/tensors/'),
        data_root=
        'data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass/',
        metainfo=dict(
            classes=(
                '1024qam',
                '128qam_cross',
                '16ask',
                '16fsk',
                '16gfsk',
                '16gmsk',
                '16msk',
                '16psk',
                '16qam',
                '2fsk',
                '2gfsk',
                '2gmsk',
                '2msk',
                '256qam',
                '32ask',
                '32psk',
                '32qam',
                '32qam_cross',
                '4ask',
                '4fsk',
                '4gfsk',
                '4gmsk',
                '4msk',
                '512qam_cross',
                '64ask',
                '64psk',
                '64qam',
                '8ask',
                '8fsk',
                '8gfsk',
                '8gmsk',
                '8msk',
                '8psk',
                'am-dsb',
                'am-dsb-sc',
                'am-lsb',
                'am-usb',
                'bpsk',
                'chirpss',
                'fm',
                'lfm-data',
                'lfm-radar',
                'ofdm-1024',
                'ofdm-1200',
                'ofdm-128',
                'ofdm-180',
                'ofdm-2048',
                'ofdm-256',
                'ofdm-300',
                'ofdm-512',
                'ofdm-600',
                'ofdm-64',
                'ofdm-72',
                'ofdm-900',
                'ook',
                'qpsk',
                'tone',
            )),
        pipeline=[
            dict(
                expected_channels=3,
                memmap_root=
                'data/torchsig_hardshort_lowsnr_stft3_memmap/memmap',
                type='LoadTensorMemmapFromCOCOStem'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'complex_stft_shape',
                    'tensor_memmap_shape',
                    'memmap_index',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file=
    'data/torchsig_hardshort_lowsnr_stft3_memmap/coco_multiclass/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/baseline_mc_rtmdet_m_20e_seed20262811'
