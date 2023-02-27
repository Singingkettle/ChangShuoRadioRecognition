_base_ = [
    '../_base_/schedules/schedule.py',
    './data_pure-csss.py',
    '../_base_/default_runtime.py',
]

num_mod = 5
# Model
model = dict(
    type='SingleHeadClassifier',

    backbone=dict(
        type='CRNet',
        in_channels=1,
        cnn_depth=3,
        rnn_depth=2,
        input_size=80,
        out_indices=(2,),
        avg_pool=(1, 8),
        rnn_mode='GRU',
    ),
    classifier_head=dict(
        type='ACMHead',
        num_classes=num_mod,
        in_features=50,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

optimizer = dict(type='Adam', lr=0.0001)
total_epochs = 2000