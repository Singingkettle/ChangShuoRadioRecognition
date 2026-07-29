"""Wave-9 Tier-A: HCGDNN — true paper MultiStep@800 / 1600ep + L2.

W8 best 62.88 used milestone=[400] @800ep. Paper schedule is milestone 800 / 1600ep.
"""

_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']

train_cfg = dict(by_epoch=True, max_epochs=1600, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[800],
    gamma=0.3,
)
