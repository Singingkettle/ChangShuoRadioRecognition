"""Wave-9 Tier-A: HCGDNN — AdamW decoupled WD on L2 MultiStep stack, 600ep."""

_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00044, weight_decay=1e-4),
)

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[300],
    gamma=0.3,
)
