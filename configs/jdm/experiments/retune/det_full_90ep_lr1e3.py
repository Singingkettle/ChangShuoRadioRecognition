# Full-data detector, 90-epoch cosine (architecture freeze; longer schedule than
# the 60ep rung). Escalation rung for closing the ideal-det gap (v1 test-only
# eval ~0.80 vs paper 0.91) — SAME model/anchors/losses/optimizer type as
# jdm-det_fft-csrd.py, only the training LENGTH (schedule + max_epochs) changes.
# Trains on the full simulate mixture; evaluate on v1 test-only via
# eval_ideal_v1_det_testonly.py.
_base_ = '../../jdm-det_fft-csrd.py'

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=90,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=90, val_interval=1)

work_dir = 'work_dirs/jdm/retune/det_full_90ep_lr1e3'
