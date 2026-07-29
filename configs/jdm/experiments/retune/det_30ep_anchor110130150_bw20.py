# Full 30-epoch detector — paper AP-bin anchors (110/130/150), bw loss ×20.
_base_ = '../../jdm-det_fft-csrd.py'

model = dict(
    head=dict(
        anchor_widths=(110.0, 130.0, 150.0),
        loss_bw=dict(type='MSELoss', loss_weight=20.0),
    ))

work_dir = 'work_dirs/jdm/retune/det_30ep_anchor110130150_bw20'
