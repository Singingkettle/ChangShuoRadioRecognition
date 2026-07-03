# JDM detection module (YOLO-style 1-D signal detector) on CSRD/CRML23.
# Paper: Xing et al., "Joint Signal Detection and Automatic Modulation
# Classification via Deep Learning", IEEE TWC 2024 (Sec. V-B, VI).
_base_ = [
    '../_base_/datasets/csrd/det-fft-csrd.py',
    '../_base_/runtimes/det.py',
]

model = dict(
    type='SignalDetector',
    backbone=dict(
        type='JDMDetectionBackbone',
        in_channels=2,
        stage_channels=(16, 32, 64, 128, 256),
        init_cfg=dict(type='Xavier', layer='Conv1d'),
    ),
    head=dict(
        type='JDMDetectionHead',
        in_channels=256,
        frame_length=1200,
        stride=8,
        anchor_widths=(100.0, 120.0, 140.0),
        ignore_iou_thr=0.5,
        loss_conf=dict(type='CrossEntropyLoss', use_sigmoid=True,
                       loss_weight=1.0),
        loss_cf=dict(type='CrossEntropyLoss', use_sigmoid=True,
                     loss_weight=1.0),
        loss_bw=dict(type='MSELoss', loss_weight=2.0),
        test_cfg=dict(score_thr=0.05, nms_iou_thr=0.45, max_per_frame=20),
    ),
)

# paper Sec. VI: Adam, lr 1e-3, 30 epochs, batch 12 (batch set in dataset base)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=30,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()
