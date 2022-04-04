_base_ = [
    '../_base_/datasets/iq-deepsig-201610A.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# Model
model = dict(
    type='DNN',
    method_name='CNN2-IQ',
    backbone=dict(
        type='CNNNet',
        depth=2,
        in_channels=1,
        out_indices=(1,),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_features=10560,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()
