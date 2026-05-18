_base_ = [
    '../../../_base_/datasets/deepsig/iq-shape-L-F-deepsig201801A.py',
    '../../../_base_/schedules/amc.py',
    '../../../_base_/runtimes/amc.py',
]

data_root = '/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2018.01A'

snr_pipeline = [
    dict(type='Transpose', orders=dict(iq=[1, 0])),
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
        type='PETCGDNN',
        frame_length=1024,
        num_classes=24,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
