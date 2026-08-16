_base_ = [
    "../../../_base_/datasets/deepsig/iq-deepsig201801A.py",
    "../../../_base_/schedules/amc.py",
    "../../../_base_/runtimes/amc.py",
]

data_root = "data/ModulationClassification/DeepSig/RadioML.2018.01A"

snr_pipeline = [
    dict(type="Reshape", shapes=dict(iq=[1, 2, 1024])),
    dict(
        type="PackInputs",
        input_key="iq",
        meta_keys=("sample_idx", "snr", "snr_label", "modulation")),
]

train_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
val_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))
test_dataloader = dict(dataset=dict(data_root=data_root, pipeline=snr_pipeline))

model = dict(
    type="SignalClassifier",
    backbone=dict(
        type="CGDNet",
        # RadioML2018.01A records contain 1024 complex samples per example.
        # CGDNet uses this value to reshape the convolutional feature map
        # before the GRU; using the 2016A value (128) causes a view mismatch.
        frame_length=1024,
        num_classes=24,
        init_cfg=[
            dict(type="Kaiming", layer="Linear", mode="fan_in"),
            dict(type="RNN", layer="GRU", gain=1),
            dict(
                type="Xavier",
                layer="Conv2d",
                distribution="uniform",
                override=dict(type="Uniform", name="cnn1", a=-0.108253, b=0.108253)),
        ],
    ),
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
