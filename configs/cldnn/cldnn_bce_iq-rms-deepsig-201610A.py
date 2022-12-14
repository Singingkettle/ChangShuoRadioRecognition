_base_ = [
        '../_base_/datasets/deepsig/rebase_modulation_label_by_snr/iq-rms-deepsig-201610A.py',
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
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_features=50,
        out_features=128,
        loss_cls=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)