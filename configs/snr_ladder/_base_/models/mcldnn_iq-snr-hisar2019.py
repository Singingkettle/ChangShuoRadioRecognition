_base_ = [
    '../../../_base_/datasets/hisar/iq-hisar2019.py',
    '../../../_base_/schedules/amc.py',
    '../../../_base_/runtimes/amc.py',
]

data_root = 'data/ModulationClassification/Hisar/HisarMod2019.1'

snr_pipeline = [
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1024])),
    dict(
        type='PackInputs',
        input_key='iq',
        meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))
val_dataloader = dict(
    batch_size=256,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))
test_dataloader = dict(
    batch_size=256,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MCLDNN',
        frame_length=1024,
        num_classes=26,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
