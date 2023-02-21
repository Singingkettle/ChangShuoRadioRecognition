dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=20, persistent_workers=True, prefetch_factor=3,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[-2]),
        ],
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 file_name='train_and_validation_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['inputs', 'targets'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[-2]),
        ],
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 file_name='test_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNR', )
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[-2]),
        ],
        pipeline=[
            dict(type='LoadConstellationFromCache', data_root=data_root,
                 file_name='test_filter_size_0.020_stride_0.020.pkl', to_float32=True),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNR', )
        ],
        save=[
            dict(type='SaveModulationPrediction', )
        ],
    ),
)
