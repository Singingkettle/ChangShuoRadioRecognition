# Hard cross-entropy baseline (the frozen model the ladder audits).
# Paper: "Do SNR-Aware Training Gains Survive a Frozen-Model Readout? A Null-Ladder Audit of Modulation Classification", under review (2026).
_base_ = [
    '../../_base_/datasets/ucsd/iq-ucsdrml22.py',
    '../../_base_/schedules/amc.py',
    '../../_base_/runtimes/amc.py',
]

data_root = 'data/ModulationClassification/UCSD/RML22'

# RML22 IQ amplitudes sit two orders of magnitude below DeepSig's; without a
# per-sample normalization the attention/GRU backbones cannot break symmetry
# and collapse to chance (documented deviation in the README).
snr_pipeline = [
    dict(type='SelfNormalize', norms=dict(iq=dict())),
    dict(type='Reshape', shapes=dict(iq=[2, 128])),
    dict(
        type='PackInputs',
        input_key='iq',
        meta_keys=('sample_idx', 'snr', 'modulation')),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='MCformer',
        fea_dim=32,
        num_classes=10,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
