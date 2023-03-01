dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4, persistent_workers=True, prefetch_factor=3,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',

        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True),
            dict(type='LoadAPFromCache', data_root=data_root, file_name='train_and_validation_ap.pkl', to_float32=True,
                 to_norm=True),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='Collect', keys=['inputs', 'targets'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromCache', data_root=data_root, file_name='test_ap.pkl', to_float32=True),
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
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromCache', data_root=data_root, file_name='test_ap.pkl', to_float32=True),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
        save=[
            dict(type='FormatSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
)
