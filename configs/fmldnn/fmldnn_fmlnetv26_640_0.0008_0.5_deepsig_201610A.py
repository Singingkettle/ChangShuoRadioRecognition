_base_ = '../_base_/default_runtime.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
        use_snr_label=True,
        snr_threshold=0.0,
        channel_mode=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        use_snr_label=True,
        snr_threshold=0.0,
        channel_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        use_snr_label=True,
        snr_threshold=0.0,
        channel_mode=True,
    ),
)

model = dict(
    type='FMLDNN',
    backbone=dict(
        type='FMLNetV26',
    ),
    classifier_head=dict(
        type='FMLHierarchicalHead',
        heads=[
            # CNN Head
            dict(
                type='AMCHead',
                num_classes=11,
                in_features=1200,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1,
                ),
            ),
            # LSTM1 Head
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
            # LSTM2 Head
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
        ]
    ),
)
train_cfg = dict()
test_cfg = dict()

total_epochs = 400
# optimizer
optimizer = dict(type='Adam', lr=0.0008)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
