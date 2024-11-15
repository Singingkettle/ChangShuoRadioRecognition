_base_ = [
    './iq-ap-deepsig201610A.py',
    './runtimes.py',
]

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0001054)
)

param_scheduler = dict(
    type='ConstantLR',
    factor=1
)

train_cfg = dict(by_epoch=True, max_epochs=3200, val_interval=1)
val_cfg = dict()
test_cfg = dict()


# model settings
model = dict(
    type='SignalClassifier',
    backbone=dict(
        type='FastMLDNN',
        num_classes=11,
        dp=0.07,
        init_cfg=dict(type='Pretrained',
                      checkpoint='./work_dirs/fastmldnn_iq-deepsig-201610A/best_accuracy_top1_epoch_648.pth',
                      prefix='backbone.')
    ),
    head=dict(
        type='FastMLDNNHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        beta=0.5,
    )
)
