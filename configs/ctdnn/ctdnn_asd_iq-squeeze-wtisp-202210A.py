_base_ = [
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'WTISPDataset'
data_root = '/home/citybuster/Data/SignalProcessing/SpecificEmitterIdentification/WTISP/202210A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=4, persistent_workers=True, prefetch_factor=3,
    train=[
        dict(
            type=dataset_type,
            ann_file='train.json',
            pipeline=[
                dict(type='LoadIQFromFile', is_squeeze=True, to_float32=True),
                dict(type='LoadAnnotations', with_mod=True, with_dev=True),
                dict(type='Collect', keys=['iqs', 'mod_labels', 'dev_labels'])
            ],
            data_root=data_root,
        ),
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadIQFromFile', is_squeeze=True, to_float32=True),
                dict(type='LoadAnnotations', with_mod=True, with_dev=True),
                dict(type='Collect', keys=['iqs', 'mod_labels', 'dev_labels'])
            ],
            data_root=data_root,
        ),
    ],
    val=dict(
        type=dataset_type,
        ann_file='train.json',
        pipeline=[
            dict(type='LoadIQFromFile', is_squeeze=True, to_float32=True),
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateTwoClassificationTaskWithSNR', )
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromFile', is_squeeze=True, to_float32=True),
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateTwoClassificationTaskWithSNR', )
        ],
        save=[
            dict(type='SaveModulationPrediction', )
        ],
    ),
)


# model
model = dict(
    type='DNN',
    method_name='CTDNN',
    backbone=dict(
        type='CTNet',
        in_channel=2,
        growth_rate=16,
        num_layers=10,
        kernel_size=11
    ),
    classifier_head=dict(
        type='ASDHead',
        amc_cls_num=5,
        sei_cls_num=5,
        in_features=162,
        out_features=256,
        loss_amc=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        loss_sei=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
