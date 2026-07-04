# Short JDM detector localization experiment.
#
# Diagnosis on the baseline detector shows AP50 is saturated while AP75 is
# limited by large bandwidth errors, especially on the ~146-bin cluster. This
# variant keeps the baseline architecture but uses empirical width anchors and
# a stronger log-bandwidth loss for a bounded 5-epoch trend check.
_base_ = '../jdm-det_fft-csrd.py'

model = dict(
    head=dict(
        anchor_widths=(96.0, 120.0, 146.0),
        loss_bw=dict(type='MSELoss', loss_weight=20.0),
    ))

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=5,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
work_dir = 'work_dirs/jdm/exp_det_anchor096146_bw20_5ep'
