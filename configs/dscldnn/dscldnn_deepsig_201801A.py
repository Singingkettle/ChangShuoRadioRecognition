_base_ = 'dscldnn_deepsig_201604C.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=1,
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

model = dict(
    type='DSCLDNN',
    backbone=dict(
        type='DSCLNet',
        in_channels=1,
        cnn_depth=3,
        rnn_depth=2,
        input_size=80,
        out_indices=(2,),
        avg_pool=(1, 8),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='DSAMCHead',
        num_classes=24,
        in_features=2500,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
