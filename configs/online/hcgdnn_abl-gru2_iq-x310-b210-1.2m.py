_base_ = '../_base_/default_runtime.py'

dataset_type = 'OnlineDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_b210_1.2m/Online'
data = dict(
    samples_per_gpu=3200,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='train.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True),
            dict(type='ChannelMode'),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['iqs', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=dict(
        type=dataset_type,
        ann_file='val.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True),
            dict(type='ChannelMode'),
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateOnlineModulationPrediction', prediction_name='HCGDNN')
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file='val.json',
        pipeline=[
            dict(type='LoadIQFromFile', to_float32=True),
            dict(type='ChannelMode'),
            dict(type='Collect', keys=['iqs'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateOnlineModulationPrediction', prediction_name='HCGDNN')
        ],
    ),
)

in_size = 100
out_size = 288
# Model
model = dict(
    type='DNN',
    method_name='HCGDNN-GRU2',
    backbone=dict(
        type='HCGNetGRU2',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='AMCHead',
        in_features=in_size,
        out_features=out_size,
        num_classes=8,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

total_epochs = 400

# Optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
