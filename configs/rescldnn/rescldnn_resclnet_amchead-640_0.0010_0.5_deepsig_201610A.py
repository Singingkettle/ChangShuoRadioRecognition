_base_ = '../_base_/default_runtime.py'

batch_size = 80
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
    ),
)

in_features = 80
num_classes = 11
model = dict(
    type='ResCLDNN',
    backbone=dict(
        type='ResCLNetV1',
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_features,
        num_classes=num_classes,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)
train_cfg = dict()
test_cfg = dict()

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # which has bug to work with the matlab.engine
        dict(type='TensorboardLoggerHook')
    ])

total_epochs = 400
# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
