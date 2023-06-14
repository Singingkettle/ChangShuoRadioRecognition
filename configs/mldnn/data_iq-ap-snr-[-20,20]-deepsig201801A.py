dataset_type = 'DeepSigDataset'
data_root = './data/ModulationClassification/DeepSig/201801A'
target_name = 'modulation'
data = dict(
    samples_per_gpu=80,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        preprocess=[
          dict(type='FilterBySNR', snr_set=[snr for snr in range(-20, 22, 2)])
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='LoadAnnotations', target_info={target_name: 'int64'}),
            dict(type='MLDNNSNRLabel'),
            dict(type='Collect', keys=['inputs', 'targets'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-20, 22, 2)])
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateMLDNN'),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='FilterBySNR', snr_set=[snr for snr in range(-20, 22, 2)])
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateMLDNN'),
        ],
        format=[
            dict(type='SaveMLDNN'),
        ],
    ),
)
