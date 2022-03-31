_base_ = [
    '../_base_/datasets/iq-deepsig-201801A.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='CLDNN2-IQ',
    backbone=dict(
        type='CRNet',
        in_channels=1,
        cnn_depth=3,
        rnn_depth=2,
        input_size=80,
        out_indices=(2,),
        avg_pool=(1, 8),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='DSAMCHead',
        num_classes=24,
        in_features=50,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()


