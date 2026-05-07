_base_ = [
    '../../../_base_/datasets/deepsig/iq-deepsig201610A.py',
    '../../../_base_/schedules/amc.py',
    '../../../_base_/runtimes/amc.py',
]

data_root = '/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10A'

snr_pipeline = [
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(
        type='PackInputs',
        input_key='iq',
        meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='CNN2',
        num_classes=11,
        init_cfg=dict(type='Xavier', layer='Conv2d'),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
