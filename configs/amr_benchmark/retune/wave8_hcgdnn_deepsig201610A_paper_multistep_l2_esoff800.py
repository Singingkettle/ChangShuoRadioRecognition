"""Wave-8 Tier-A: HCGDNN — extend W7 best (62.15/91.89) to paper-scale 800ep."""

_base_ = ['./wave7_hcgdnn_deepsig201610A_paper_multistep_l2_esoff400.py']

train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)

param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[400],
    gamma=0.3,
)
