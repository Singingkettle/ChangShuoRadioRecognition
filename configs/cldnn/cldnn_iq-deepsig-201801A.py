_base_ = [
    '../_base_/datasets/iq-deepsig-201801A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='CLDNN-IQ',
    backbone=dict(
        type='CRNet',
        in_channels=1,
        cnn_depth=4,
        rnn_depth=1,
        input_size=80,
        out_indices=(3,),
        avg_pool=(1, 8),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='ClassificationHead',
        num_classes=24,
        in_features=50,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
