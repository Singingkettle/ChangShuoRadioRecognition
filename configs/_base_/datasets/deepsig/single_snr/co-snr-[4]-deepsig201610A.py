dataset_type = 'DeepSigDataset'
data_root = './data/ModulationClassification/DeepSig/201610A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[4]),
        ],
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 file_name='train_and_validation_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='Collect', keys=['inputs', 'targets'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[4]),
        ],
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 file_name='test_filter_size_0.020_stride_0.020.pkl', to_float32=True),
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
        preprocess=[
            dict(type='FilterBySNR', snr_set=[4]),
        ],
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 file_name='test_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='Collect', keys=['inputs'])
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
