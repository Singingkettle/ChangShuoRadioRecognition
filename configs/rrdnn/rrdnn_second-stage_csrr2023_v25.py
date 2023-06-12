_base_ = [
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'DeepSigDataset'
data_root = './data/ChangShuo/v25'
target_name = 'modulation'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='ts-train_and_validation.json',
        # ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='ts-train_and_validation_iq.pkl',
                 to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='Collect', keys=['inputs', 'targets'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
        format=[
            dict(type='FormatSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file='ts-test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='ts-test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='ts-test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='ts-test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
    ),
)

is_det = False

in_size = 100
out_size = 288
model = dict(
    type='HCGDNN',
    backbone=dict(
        type='HCGNetCNN',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=6,
        in_size=in_size,
        out_size=out_size,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1
        ),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    )
)

runner = dict(type='EpochBasedRunner', max_epochs=100)
# Optimizer
# for flops calculation
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.00005)
# learning policy
lr_config = dict(policy='fixed',)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
