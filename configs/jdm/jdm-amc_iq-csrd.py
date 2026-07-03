# JDM modulation-classification module on CSRD/CRML23 baseband crops.
# Paper: Xing et al., "Joint Signal Detection and Automatic Modulation
# Classification via Deep Learning", IEEE TWC 2024 (Sec. V-C, VI).
_base_ = [
    '../_base_/datasets/csrd/iq-baseband-csrd.py',
    '../_base_/runtimes/amc.py',
]

model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='JDMClassificationBackbone',
        num_classes=5,
        dropout_rate=0.5,
        init_cfg=dict(type='Xavier', layer='Conv2d'),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)

# paper Sec. VI: AdamW, lr 1e-3, weight decay 5e-5, 60 epochs, batch 32
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=5e-5),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=60,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
val_cfg = dict()
test_cfg = dict()
