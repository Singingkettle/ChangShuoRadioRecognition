dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/2010801A'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        augment=[
            dict(type='FilterBySNR', low_snr=-8, high_snr=20),
            dict(type='MLDNNSNRLabel'),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='train_and_validation_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['iqs', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        augment=[
            dict(type='FilterBySNR', low_snr=-8, high_snr=20),
            dict(type='MLDNNSNRLabel'),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['iqs', ])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPredictionOfHCGDNN', merge=dict(type='Optimization'))
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        augment=[
            dict(type='FilterBySNR', low_snr=-8, high_snr=20),
            dict(type='MLDNNSNRLabel'),
        ],
        pipeline=[
            dict(type='LoadIQFromCache', data_root=data_root, filename='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['iqs', ])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPredictionOfHCGDNN', merge=dict(type='Optimization'))
        ],
        save=[
            dict(type='SaveModulationPredictionOfHCGDNN', merge=dict(type='Optimization'))
        ],
    ),
)
