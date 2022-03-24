_base_ = [
    '../_base_/datasets/ap-deepsig-201610A.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='CRNN',
    is_iq=True,
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
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 400

# Optimizer
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
