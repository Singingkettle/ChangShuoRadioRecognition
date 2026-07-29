"""Paper-recipe siege: HCGDNN @ 10A — MultiStep @200 / 400ep, ES off.

Shorter scaled paper schedule (half of the 800ep scaled recipe).
Adam 4.4e-4; HCGDNNHook retained; no EarlyStopping.
"""

_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.00044))

train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[200],
    gamma=0.3,
)

custom_hooks = [
    dict(type='HCGDNNHook'),
]
