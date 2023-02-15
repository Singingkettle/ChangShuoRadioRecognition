dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=320,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        preprocess=[
            dict(type='MLDNNSNRLabel'),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='train_and_validation_iq.pkl', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='LoadAnnotations', with_snr=True),
            dict(type='Collect', keys=['iqs', 'aps', 'mod_labels', 'snr_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='MLDNNSNRLabel'),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNR'),
            dict(type='EvaluateSNRPrediction')
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        preprocess=[
            dict(type='MLDNNSNRLabel'),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='LoadAPFromIQ'),
            dict(type='Collect', keys=['inputs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateClassificationWithSNR'),
            dict(type='EvaluateSNRPrediction')
        ],
        save=[
            dict(type='SaveModulationPrediction'),
            dict(type='SaveSNRPrediction')
        ],
    ),
)
