dataset_type = 'CSRRDataset'
data_root = '/home/citybuster/Data/SignalProcessing/SignalRecognition/ChangShuo/CSRR2023'
target_name = 'modulation'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='LoadCSRRTrainAnnotations'),
            dict(type='Collect', keys=['inputs', 'targets', 'input_metas'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs', 'input_metas'])
        ],
        data_root=data_root,
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs',])
        ],
        data_root=data_root,
    ),
)
