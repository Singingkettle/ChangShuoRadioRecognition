"""Wave-9 Tier-A: FastMLDNN — exact paper MultiStep [20,80,400,600,760] + L2, 800ep.

W7 used scaled milestones for 500ep; W4 channel-lr used exact schedule but NO L2.
Combine exact paper drops with the L2 near-miss stack (best ~60.89).
"""

_base_ = ['./wave6_fastmldnn_deepsig201610A_paper_fixedlr_l2_esoff500.py']

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.00044, weight_decay=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[20, 80, 400, 600, 760],
    gamma=0.3,
)
