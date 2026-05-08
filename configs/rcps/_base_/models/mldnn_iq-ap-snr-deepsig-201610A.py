_base_ = ['../../../mldnn/mldnn_iq-ap-deepsig201610A.py']

data_root = '/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10A'

train_pipeline = [
    dict(type='MLDNNSNRLabel'),
    dict(type='MLDNNIQToAP'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='Reshape', shapes=dict(ap=[1, 2, 128])),
    dict(
        type='PackMultiTaskInputs',
        multi_task_fields=['gt_label'],
        input_key=['iq', 'ap'],
        meta_keys=('sample_idx', ('snr', 'snr_db'), 'snr_label', 'modulation'),
        task_handlers=dict(amc=dict(type='PackInputs'), snr=dict(type='PackInputs')),
    ),
]

pipeline = [
    dict(type='MLDNNIQToAP'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='Reshape', shapes=dict(ap=[1, 2, 128])),
    dict(type='PackInputs', input_key=['iq', 'ap'], meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=pipeline))
