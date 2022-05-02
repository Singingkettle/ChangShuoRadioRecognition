_base_ = [
    './data_iq-channel-snr-[-8,30]-deepsig-201610A.py',
    '../_base_/default_runtime.py'
]

in_size = 100
# Model
model = dict(
    type='DNN',
    method_name='SEDNN',
    backbone=dict(
        type='HCGNetGRU2',
        input_size=in_size,
    ),
    classifier_head=dict(
        type='SEDNNHead',
        num_snr=15,
        in_features=in_size,
        mod_out_features=64,
        snr_out_features=64,
        num_mod=24,
        snrs=['{}'.format(snr) for snr in range(-8, 22, 2)],
        loss_mod=dict(
            type='CrossEntropyLoss',
            loss_weight=1,
        ),
        loss_snr=dict(
            type='CrossEntropyLoss',
            loss_weight=0.5,
        ),
        loss_merge=dict(
            type='NLLLoss',
            loss_weight=0.1,
        )
    ),
)

# Optimizer
optimizer = dict(type='AdamW', lr=0.0004)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
