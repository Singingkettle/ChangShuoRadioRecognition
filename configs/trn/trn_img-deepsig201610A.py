_base_ = [
    './data_img-deepsig201610A.py',
    './schedule.py',
    '../_base_/default_runtime.py'
]

# Some of the parameters presented in the original paper is not work well, so we change some of them.
# Of course, we ask the source code help from the author, but there is no replay.

# Model
model = dict(
    type='TRN',
    backbone=dict(
        type='TRNet',
        image_size=(2, 128),
        patch_size=(2, 8)
    ),
    classifier_head=dict(
        type='VitHead',
        num_classes=11,
        in_size=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
    ),
)
