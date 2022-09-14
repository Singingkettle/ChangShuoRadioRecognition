_base_ = [
    '../_base_/datasets/iq-deepsig-201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='CNN4-IQ',
    vis_fea=True,
    backbone=dict(
        type='CNNNet',
        depth=4,
        in_channels=1,
        out_indices=(3,),
    ),
    classifier_head=dict(
        type='VisHead',
        num_classes=11,
        in_features=10880,
        out_features=2,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
