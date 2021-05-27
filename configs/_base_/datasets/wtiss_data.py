dataset_type = 'WTISSV2Dataset'
data_root = '/home/citybuster/Data/SignalProcessing/SignalSeparation/WTISS/qpsk_16qam/real'
split_length = 128
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        split_length=split_length,
        data_root=data_root,
        data_name='train_data_20_0.10.mat',
    ),
    val=dict(
        type=dataset_type,
        split_length=split_length,
        data_root=data_root,
        data_name='test_data_20_0.10.mat',
    ),
    test=dict(
        type=dataset_type,
        split_length=split_length,
        data_root=data_root,
        data_name='test_data_20_0.10.mat',
    ),
)
evaluation = dict(interval=1)
