"""Paper-recipe siege: FastMLDNN @ 10A — paper MultiStep + β=0.5, no L2.

Aligns to configs/fastmldnn/paper/ channel full recipe under CSRR 50/10/40:
  - beta/balance = 0.5 (center-distance multi-loss)
  - backbone dp = 0.07
  - MultiStepLR gamma=0.3, milestones scaled from paper [20,80,400,600,760]
  - Adam lr = 4.4e-4 (paper iq-ap headline LR; channel stage used 1.054e-4)
  - ES off, 400 epochs (paper iq-ap runner.max_epochs)
  - No IQ SelfNormalize (paper pipelines have no per-sample L2)

Architecture frozen.
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

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.00044))

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[20, 80, 200, 300, 380],
    gamma=0.3,
)

custom_hooks = []
