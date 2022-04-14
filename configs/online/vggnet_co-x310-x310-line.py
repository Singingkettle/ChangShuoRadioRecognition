_base_ = '../_base_/default_runtime.py'

dataset_type = 'OnlineDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_x310_line/Online'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='train.json',
        pipeline=[
            dict(type='LoadConstellationFromIQFile', to_float32=True),
            dict(type='ChannelMode'),
            dict(type='LoadAnnotations'),
            dict(type='Collect', keys=['cos', 'mod_labels'])
        ],
        data_root=data_root,
    ),
    val=[
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadConstellationFromIQFile', to_float32=True),
                dict(type='ChannelMode'),
                dict(type='Collect', keys=['cos'])
            ],
            data_root='/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_x310_0.25m/Online',
            evaluate=[
                dict(type='EvaluateOnlineModulationPrediction')
            ],
        ),
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadConstellationFromIQFile', to_float32=True),
                dict(type='ChannelMode'),
                dict(type='Collect', keys=['cos'])
            ],
            data_root='/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_x310_1.2m/Online',
            evaluate=[
                dict(type='EvaluateOnlineModulationPrediction')
            ],
        ),
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadConstellationFromIQFile', to_float32=True),
                dict(type='ChannelMode'),
                dict(type='Collect', keys=['cos'])
            ],
            data_root='/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_x310_3m/Online',
            evaluate=[
                dict(type='EvaluateOnlineModulationPrediction')
            ],
        ),
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadConstellationFromIQFile', to_float32=True),
                dict(type='ChannelMode'),
                dict(type='Collect', keys=['cos'])
            ],
            data_root='/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_b210_x310_line/Online',
            evaluate=[
                dict(type='EvaluateOnlineModulationPrediction')
            ],
        ),
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadConstellationFromIQFile', to_float32=True),
                dict(type='ChannelMode'),
                dict(type='Collect', keys=['cos'])
            ],
            data_root='/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_x310_line/Online',
            evaluate=[
                dict(type='EvaluateOnlineModulationPrediction')
            ],
        ),
        dict(
            type=dataset_type,
            ann_file='val.json',
            pipeline=[
                dict(type='LoadConstellationFromIQFile', to_float32=True),
                dict(type='ChannelMode'),
                dict(type='Collect', keys=['cos'])
            ],
            data_root='/home/citybuster/Data/SignalProcessing/ModulationClassification/Online/ModulationClassification_x310_b210_line/Online',
            evaluate=[
                dict(type='EvaluateOnlineModulationPrediction')
            ],
        ),
    ],
    test=dict(
        type=dataset_type,
        ann_file='val.json',
        pipeline=[
            dict(type='LoadConstellationFromIQFile', to_float32=True),
            dict(type='ChannelMode'),
            dict(type='Collect', keys=['cos'])
        ],
        data_root=data_root,
        evaluate=[
            dict(type='EvaluateOnlineModulationPrediction')
        ],
    ),
)

model = dict(
    type='DNN',
    method_name='VGGNet',
    backbone=dict(
        type='VGGNetCO',
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=8,
        in_features=512,
        out_features=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
    ),
)

# optimizer
optimizer = dict(type='Adam', lr=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
