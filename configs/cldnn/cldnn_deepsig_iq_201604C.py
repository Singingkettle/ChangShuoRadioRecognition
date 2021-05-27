_base_ = '../_base_/default_runtime.py'

# Dataset
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201604C'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        ap=False,
        iq=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        ap=False,
        iq=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        ap=False,
        iq=True,
    ),
)

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
