# Full-length counterpart of the bounded localization experiment.
#
# Launch this only if the 5-epoch run improves validation AP75/mAP.
_base_ = '../jdm-det_fft-csrd.py'

model = dict(
    head=dict(
        anchor_widths=(96.0, 120.0, 146.0),
        loss_bw=dict(type='MSELoss', loss_weight=20.0),
    ))

work_dir = 'work_dirs/jdm/exp_det_anchor096146_bw20'
