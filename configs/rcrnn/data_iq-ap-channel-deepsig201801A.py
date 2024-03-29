dataset_type = 'DeepSigDataset'
data_root = './data/ModulationClassification/DeepSig/201801A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True,
                 is_squeeze=True),
            dict(type='LoadDerivativeFromIQ', to_float32=True, is_squeeze=True),
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
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True,
                 is_squeeze=True),
            dict(type='LoadDerivativeFromIQ', to_float32=True, is_squeeze=True),
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
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True,
                 is_squeeze=True),
            dict(type='LoadDerivativeFromIQ', to_float32=True, is_squeeze=True),
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
