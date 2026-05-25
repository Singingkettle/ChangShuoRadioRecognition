_base_ = [
    '../../../_base_/datasets/deepsig/iq-shape-L-F-deepsig201610B.py',
    '../../../_base_/schedules/amc.py',
    '../../../_base_/runtimes/amc.py',
]

data_root = '/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10B'

snr_pipeline = [
    dict(type='LoadIQFromFile'),
    dict(type='Transpose', orders=dict(iq=[1, 0])),
    dict(type='PackInputs', input_key='iq', meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline, cache=True))

model = dict(type='SignalClassifier', backbone=dict(type='PETCGDNN', num_classes=10), head=dict(type='ClsHead', loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
