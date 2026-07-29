"""Paper-recipe siege: FastMLDNN @ 10A — channel-stage LR + paper MultiStep.

Closest to configs/fastmldnn/paper/fastmldnn_iq-ap-channel-deepsig201610A.py:
  - Adam lr = 1.054e-4
  - MultiStepLR gamma=0.3, milestones=[20, 80, 400, 600, 760]
  - max_epochs=800 (paper 3200 capped for wall-clock; drops through ep 760)
  - beta=0.5, dp=0.07, Xavier+TruncNormal, ES off, no IQ L2

Architecture frozen. Pretrained init from paper stage1 omitted (no checkpoint).
"""

_base_ = ['../../fastmldnn/fastmldnn_iq-ap-deepsig-201610A.py']

model = dict(
    backbone=dict(
        dp=0.07,
        init_cfg=[
            dict(type='Xavier', layer='Conv1d', distribution='uniform'),
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ],
    ),
    head=dict(beta=0.5),
)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0001054))

train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[20, 80, 400, 600, 760],
    gamma=0.3,
)

custom_hooks = []
