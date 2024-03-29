_base_ = [
    './data_two-stage-csss.py',
    '../_base_/default_runtime.py',
]

num_band = 5
num_mod = 5
# Model
model = dict(
    type='SSNNTwoStage',
    band_net=dict(
        type='BaseClassifier',

        backbone=dict(
            type='CNNNet',
            depth=4,
            in_channels=1,
            out_indices=(3,),
            avg_pool=(1, 8),
        ),
        classifier_head=dict(
            type='AMCHead',
            num_classes=num_band,
            in_size=10240,
            out_size=128,
            loss_cls=dict(
                type='BinaryCrossEntropyLoss',
                loss_weight=1.0,
            ),
        ),
    ),
    mod_net=dict(
        type='BaseClassifier',

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
            type='AMCHead',
            num_classes=num_mod,
            in_size=50,
            out_size=128,
            loss_cls=dict(
                type='CrossEntropyLoss',
                loss_weight=1.0,
            ),
        ),
    ),
    num_band=num_band,
    num_mod=num_mod,
)

total_epochs = 1600
# optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.3,
    step=[300, 800, 1200]
)
