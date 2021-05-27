_base_ = "cnn2_deepsig_iq_201604C.py"

# Dataset
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=128,
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
    type='CNN2',
    is_iq=False,
    backbone=dict(
        type='CNNNet',
        depth=2,
        in_channels=1,
        out_indices=(1,),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_features=10560,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
