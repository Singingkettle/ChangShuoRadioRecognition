_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=60,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        use_snr_label=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        use_snr_label=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        use_snr_label=True,
    ),
)

model = dict(
    type='MLCLDNN',
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
        type='MLAMCHead',
        heads=[
            # Snr Head
            dict(
                type='AMCHead',
                num_classes=2,
                in_features=50,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=0.2,
                ),
            ),
            # Low Head
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
            # High Head
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
# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
