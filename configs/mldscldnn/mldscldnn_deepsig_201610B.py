_base_ = 'mldscldnn_deepsig_201604C.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610B'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
        use_snr_label=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        use_snr_label=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        iq=True,
        ap=True,
        use_snr_label=True,
    ),
)

model = dict(
    type='MLDSCLDNN',
    backbone=dict(
        type='DSCLNet',
        in_channels=1,
        cnn_depth=3,
        rnn_depth=2,
        input_size=80,
        out_indices=(2,),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='MLAMCHead',
        heads=[
            # Snr Head
            dict(
                type='AMCHead',
                num_classes=2,
                in_features=2500,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=0.2,
                ),
            ),
            # Low Head
            dict(
                type='AMCHead',
                num_classes=10,
                in_features=2500,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=0.4,
                ),
            ),
            # High Head
            dict(
                type='AMCHead',
                num_classes=10,
                in_features=2500,
                out_features=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=0.4,
                ),
            ),
            # Merge Head
            dict(
                type='MergeAMCHead',
                loss_cls=dict(
                    type='CustomNLLLoss',
                    loss_weight=1.0,
                ),
            ),
        ]
    ),
)
