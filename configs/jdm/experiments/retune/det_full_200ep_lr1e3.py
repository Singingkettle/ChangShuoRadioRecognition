# Full-data detector, 200-epoch cosine — next escalation rung after 120ep.
# Same model/anchors/losses/optimizer as jdm-det_fft-csrd.py; only the schedule
# LENGTH changes (ladder: 30 -> 60 -> 90 -> 120 -> 200). Motivation: det120 is
# still the ideal-det champion (0.824 voted) and every fine-tune variant
# (bw40, EMA, SWA) degraded from its peak — the remaining lever inside the
# paper narrative is simply a longer cosine run.
_base_ = './det_full_120ep_lr1e3.py'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=200,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=2)

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best='detection/mAP'))

work_dir = 'work_dirs/jdm/retune/det_full_200ep_lr1e3'
