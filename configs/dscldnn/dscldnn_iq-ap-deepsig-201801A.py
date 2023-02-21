_base_ = [
    './data_iq-ap-deepsig-201801A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='SingleHeadClassifier',
    method_name='DSCLDNN',
    backbone=dict(
        type='DSCLNet',
        in_channels=1,
        cnn_depth=3,
        rnn_depth=2,
        input_size=80,
        out_indices=(2,),
        avg_pool=(1, 8),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='DSCLDNNHead',
        num_classes=24,
        in_features=2500,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
