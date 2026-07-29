"""Paper-recipe siege: HCGDNN @ 10A — MultiStep @400 / 800ep, ES off.

Aligns to configs/hcgdnn/original/schedule.py under CSRR 50/10/40:
  - Adam lr = 4.4e-4
  - MultiStepLR gamma=0.3 (paper step=[800] on 1600ep → scaled half)
  - ES off; keep HCGDNNHook for fusion-weight learning
  - Do not raise LR (prior lr1e3 retunes hurt)

Architecture frozen.
"""

_base_ = ['../../hcgdnn/hcgdnn_iq-deepsig-201610A.py']

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.00044))

train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[400],
    gamma=0.3,
)

# Keep HCGDNNHook; drop EarlyStoppingHook (paper has none).
custom_hooks = [
    dict(type='HCGDNNHook'),
]
