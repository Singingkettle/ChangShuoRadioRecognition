dataset_type = 'CSRRDataset'
data_root = './data/ChangShuo/CSRR2023'
target_name = 'modulation'
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='train_and_validation.json',
        # ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='train_and_validation_iq.pkl', to_float32=True),
            # dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='LoadCSRRTrainAnnotations'),
            dict(type='Collect', keys=['inputs', 'targets', 'input_metas', 'file_name', 'image_id'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
        format=[
            dict(type='FormatSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs', 'input_metas', 'file_name', 'image_id'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateSingleHeadClassifierWithSNR', target_name=target_name)
        ],
        format=[
            dict(type='FormatSingleHeadClassifierWithSNR', target_name=target_name)
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='test.json',
        pipeline=[
            dict(type='LoadFFTofCSRR', data_root=data_root, file_name='test_iq.pkl', to_float32=True),
            dict(type='ChannelMode', ),
            dict(type='Collect', keys=['inputs', 'input_metas', 'file_name', 'image_id'])
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
