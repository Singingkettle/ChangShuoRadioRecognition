_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
)

# Model
model = dict(
    type='HCLDNN',
    is_iq=True,
    backbone=dict(
        type='HCLNetV1',
    ),
    classifier_head=dict(
        type='HAMCHead',
        loss_prefix=['cnn', 'gru1', 'gru2', 'gru3'],
        heads=[
            # Snr Head
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=9760,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=0.2,
                ),
            ),
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=100,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=50,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=50,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
        ]
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 800

# Optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
