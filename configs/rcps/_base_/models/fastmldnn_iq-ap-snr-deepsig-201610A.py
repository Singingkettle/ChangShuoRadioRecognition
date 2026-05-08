_base_ = ['../../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

data_root = '/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2016.10A'

train_pipeline = [
    dict(type='MLDNNIQToAP'),
    dict(type='Reshape', shapes=dict(iq=[2, 128])),
    dict(type='Reshape', shapes=dict(ap=[2, 128])),
    dict(type='PackInputs', input_key=['iq', 'ap'], meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

pipeline = [
    dict(type='MLDNNIQToAP'),
    dict(type='Reshape', shapes=dict(iq=[2, 128])),
    dict(type='Reshape', shapes=dict(ap=[2, 128])),
    dict(type='PackInputs', input_key=['iq', 'ap'], meta_keys=('sample_idx', 'snr', 'snr_label', 'modulation')),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=pipeline))
