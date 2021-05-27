_base_ = [
    '../_base_/datasets/wtiss_data.py',
    '../_base_/models/tcnn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

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
        data_name='train_data_20_0.00.mat',
    ),
    val=dict(
        type=dataset_type,
        split_length=split_length,
        data_root=data_root,
        data_name='test_data_20_0.00.mat',
    ),
    test=dict(
        type=dataset_type,
        split_length=split_length,
        data_root=data_root,
        data_name='test_data_20_0.00.mat',
    ),
)

total_epochs = 36
