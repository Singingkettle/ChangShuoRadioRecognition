_base_ = [
    '../_base_/schedules/schedule.py',
    './data_single-stage-csss.py',
    '../_base_/default_runtime.py',
]

num_mod = 5
# Model
model = dict(
    type='SSNNSingleStage',
    method_name='SSNN',
    backbone=dict(
        type='SSNet',
    ),
    classifier_head=dict(
        type='SSHead',
        num_classes=num_mod + 1,
        in_features=192,
        out_features=128,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1.0,
        ),
    ),
    num_mod=num_mod + 1
)

optimizer = dict(type='Adam', lr=0.001)
total_epochs = 2000
