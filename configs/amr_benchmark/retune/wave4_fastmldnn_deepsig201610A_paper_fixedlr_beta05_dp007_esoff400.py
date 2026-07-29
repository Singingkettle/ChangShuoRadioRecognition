"""Paper-recipe siege: FastMLDNN @ 10A — paper iq-ap constant-LR regime.

configs/fastmldnn/paper/fastmldnn_iq-ap-deepsig201610A.py sets max_epochs=400
with MultiStep milestones [800,1200] — drops never fire → effective fixed LR.
Adds paper multi-loss β=0.5 + channel dp=0.07 (architecture freeze).
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
    type='ConstantLR',
    factor=1.0,
)

custom_hooks = []
