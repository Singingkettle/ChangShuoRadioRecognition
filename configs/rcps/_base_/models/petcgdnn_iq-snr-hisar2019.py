_base_ = [
    '../../../_base_/datasets/hisar/iq-shape-L-F-hisar2019.py',
    '../../../_base_/schedules/amc.py',
    '../../../_base_/runtimes/amc.py',
]

data_root = '/home/citybuster/Data/WirelessRadio/data/ModulationClassification/Hisar/HisarMod2019.1'

snr_pipeline = [
    dict(type='LoadIQFromFile'),
    dict(type='Transpose', orders=dict(iq=[1, 0])),
    dict(type='PackInputs', input_key='iq', meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))

model = dict(type='SignalClassifier', backbone=dict(type='PETCGDNN', frame_length=1024, num_classes=26), head=dict(type='ClsHead', loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
