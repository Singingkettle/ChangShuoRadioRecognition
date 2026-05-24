_base_ = [
    "../../../fastmldnn/iq-ap-deepsig201801A.py",
    "../../../_base_/schedules/amc.py",
    "../../../_base_/runtimes/amc.py",
]

data_root = "/home/citybuster/Data/WirelessRadio/data/ModulationClassification/DeepSig/RadioML.2018.01A"

snr_train_pipeline = [
    dict(type="MLDNNIQToAP"),
    dict(type="Reshape", shapes=dict(iq=[2, 1024])),
    dict(type="Reshape", shapes=dict(ap=[2, 1024])),
    dict(
        type="PackInputs",
        input_key=["iq", "ap"],
        meta_keys=("sample_idx", "snr", "snr_label", "modulation")),
]

snr_pipeline = [
    dict(type="MLDNNIQToAP"),
    dict(type="Reshape", shapes=dict(iq=[2, 1024])),
    dict(type="Reshape", shapes=dict(ap=[2, 1024])),
    dict(
        type="PackInputs",
        input_key=["iq", "ap"],
        meta_keys=("sample_idx", "snr", "snr_label", "modulation")),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_train_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))

model = dict(
    type="SignalClassifier",
    backbone=dict(type="FastMLDNN", num_classes=24),
    head=dict(
        type="FastMLDNNHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        beta=0,
    ),
)
