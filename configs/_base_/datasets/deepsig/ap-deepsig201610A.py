dataset_type = 'DeepSigDataset'
data_root = './data/ModulationClassification/DeepSig/201610A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadAPFromCache', data_root=data_root, file_name='train_and_validation_ap.pkl', to_float32=True),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='Collect', keys=[{'inputs': 'aps'}, 'targets'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadAPFromCache', data_root=data_root, file_name='test_ap.pkl', to_float32=True),
            dict(type='Collect', keys=[{'inputs': 'aps'}])
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
            dict(type='LoadAPFromCache', data_root=data_root, file_name='test_ap.pkl', to_float32=True),
            dict(type='Collect', keys=[{'inputs': 'aps'}])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
        format=[
            dict(type='FormatSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
)
