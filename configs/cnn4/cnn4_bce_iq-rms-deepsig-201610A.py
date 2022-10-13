_base_ = [
    '../_base_/datasets/deepsig/rebase_modulation_label_by_snr/iq-rms-deepsig-201610A.py',
    '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='DNN',
    method_name='CNN4-IQ',
    backbone=dict(
        type='CNNNet',
        depth=4,
        in_channels=1,
        out_indices=(3,),
    ),
    classifier_head=dict(
        type='AMCHead',
        num_classes=11,
        in_features=10880,
        out_features=128,
        loss_cls=dict(
            type='BinaryCrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
