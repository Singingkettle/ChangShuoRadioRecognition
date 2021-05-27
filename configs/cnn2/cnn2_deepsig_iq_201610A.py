_base_ = "cnn2_deepsig_iq_201604C.py"

# Dataset
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_and_val.json',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
    ),
)
