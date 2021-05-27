_base_ = 'cldnn_deepsig_iq_201604C.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
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
        avg_pool=(1, 8),
        rnn_mode='LSTM',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=24,
        in_features=50,
        out_features=128,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
