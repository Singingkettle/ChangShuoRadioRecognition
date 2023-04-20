_base_ = [
    '../_base_/datasets/doctorhe/fft-doctorhe-case4.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='CNN4',
    backbone=dict(
        type='CNNNet',
        depth=4,
        in_channels=1,
        out_indices=(3,),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=6,
        in_size=10880,
        out_size=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
lr_config = dict(policy='fixed')
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=100)