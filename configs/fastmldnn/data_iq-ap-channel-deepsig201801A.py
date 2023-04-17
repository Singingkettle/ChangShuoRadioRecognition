dataset_type = 'DeepSigDataset'
data_root = '/home/xinghuijun/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True, is_squeeze=True),
            dict(type='LoadAPFromIQ', to_float32=True, is_squeeze=True),
            dict(type='LoadSNRs'),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='Collect', keys=['inputs', 'targets', 'snrs'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True, is_squeeze=True),
            dict(type='LoadAPFromIQ', to_float32=True, is_squeeze=True),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True, is_squeeze=True),
            dict(type='LoadAPFromIQ', to_float32=True, is_squeeze=True),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
        format=[
            dict(type='FormatSingleHeadClassifierWithSNR', )
        ],
    ),
)
