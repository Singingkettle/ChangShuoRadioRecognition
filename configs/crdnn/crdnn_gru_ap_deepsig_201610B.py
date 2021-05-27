_base_ = 'crdnn_gru_ap_deepsig_201604C.py'

dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610B'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
        ap=True,
        iq=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        ap=True,
        iq=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        ap=True,
        iq=False,
    ),
)

# Model
model = dict(
    type='CRNN',
    is_iq=False,
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
        type='DSAMCHead',
        num_classes=10,
        in_features=50,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
