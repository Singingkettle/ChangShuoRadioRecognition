_base_ = [
    '../_base_/datasets/deepsig/iq-deepsig201610A.py',
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
        type='FastMLDNNHead',
        in_size=50,
        out_size=256,
        num_classes=11,
        is_reg=False,
    )
)