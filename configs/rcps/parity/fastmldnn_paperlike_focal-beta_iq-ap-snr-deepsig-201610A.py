_base_ = ["../_base_/models/fastmldnn_iq-ap-snr-deepsig-201610A.py"]

# Paper-parity diagnostic for FastMLDNN. The maintained hard-ce config uses
# dp=0.5, beta=0 and plain CE, which under-reproduces the published setting.
# This file aligns the main public/paper knobs: low dropout, focal loss,
# class-distance regularization, and the paper learning-rate schedule.

work_dir = "/home/citybuster/Data/RCPS/work_dirs/amc/deepsig201610A/fastmldnn_paperlike_focal-beta"

model = dict(
    backbone=dict(dp=0.07),
    head=dict(
        loss=dict(type="FocalLoss", loss_weight=1.0),
        beta=0.5,
    ),
)

optim_wrapper = dict(
    optimizer=dict(type="Adam", lr=0.0001054),
)

param_scheduler = dict(
    type="MultiStepLR",
    by_epoch=True,
    milestones=[20, 80, 400, 600, 760],
    gamma=0.3,
)
