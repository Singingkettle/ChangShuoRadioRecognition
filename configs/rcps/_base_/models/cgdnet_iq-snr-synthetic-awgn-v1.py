_base_ = [
    '../../../_base_/datasets/deepsig/iq-deepsig201610A.py',
    '../../../_base_/schedules/amc.py',
    '../../../_base_/runtimes/amc.py',
]

data_root = '/home/citybuster/Data/RCPS/processed/synthetic_awgn_amc_v1'

snr_pipeline = [
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(
        type='PackInputs',
        input_key='iq',
        meta_keys=(
            'sample_idx',
            'global_sample_idx',
            'snr',
            'snr_label',
            'modulation',
            'clean_id',
            'clean_file_name',
            'channel_type',
            'has_clean_signal',
            'seed',
        )),
]

train_dataloader = dict(batch_size=400, num_workers=4, dataset=dict(data_root=data_root, pipeline=snr_pipeline))
val_dataloader = dict(batch_size=400, num_workers=4, dataset=dict(data_root=data_root, pipeline=snr_pipeline))
test_dataloader = dict(batch_size=400, num_workers=4, dataset=dict(data_root=data_root, pipeline=snr_pipeline))

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='CGDNet',
        num_classes=11,
        init_cfg=[
            dict(type='Kaiming', layer='Linear', mode='fan_in'),
            dict(type='RNN', layer='GRU', gain=1),
            dict(
                type='Xavier',
                layer='Conv2d',
                distribution='uniform',
                override=dict(type='Uniform', name='cnn1', a=-0.108253, b=0.108253)),
        ],
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
