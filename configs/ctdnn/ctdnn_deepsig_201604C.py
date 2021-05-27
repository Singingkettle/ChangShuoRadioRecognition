_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201604C'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        channel_mode=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        channel_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        channel_mode=True,
    ),
)

model = dict(
    type='CTDNN',
    backbone=dict(
        type='CTNet',
        in_channels=2,
        cnn_depth=3,
        tnn_depth=2,
        sequence_length=128,
        out_indices=(2,),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_features=80,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 800
# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
