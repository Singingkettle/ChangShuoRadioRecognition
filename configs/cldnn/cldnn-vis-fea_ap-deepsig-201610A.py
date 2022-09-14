_base_ = [
    '../_base_/datasets/ap-deepsig-201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='CLDNN-AP',
    vis_fea=True,
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
        type='VisHead',
        num_classes=11,
        in_features=50,
        out_features=3,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
