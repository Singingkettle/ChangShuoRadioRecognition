dataset_type = 'DeepSigDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201801A'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=12,
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
            dict(type='LoadAnnotations', with_snr=True),
            dict(type='Collect', keys=['iqs', 'mod_labels', 'snr_labels'])
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
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPrediction', ),
            dict(type='EvaluateSNRPrediction')
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
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateModulationPrediction', ),
            dict(type='EvaluateSNRPrediction')
        ],
        save=[
            dict(type='SaveModulationPrediction'),
            dict(type='SaveSNRPrediction')
        ],
    ),
)
