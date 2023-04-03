_base_ = [
    '../_base_/datasets/ucsd/iq-ucsdrml22.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='CGDNN2',
    backbone=dict(
        type='CRNet',
        in_channels=1,
        cnn_depth=3,
        rnn_depth=2,
        input_size=80,
        out_indices=(2,),
        rnn_mode='GRU',
    ),
    classifier_head=dict(
        type='DSCLDNNHead',
        num_classes=10,
        in_size=50,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

optimizer = dict(type='Adam', lr=0.0001)
evaluation = dict(interval=1)