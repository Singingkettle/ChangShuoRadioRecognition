# Ideal-protocol detector — paper AP-bin anchors (110/130/150) + bw×20, CSRD v1 only.
# Next step after det_ideal_v1_30ep (val mAP 0.3945@ep7 / test 0.3850) toward paper ~0.91.
_base_ = './det_ideal_v1_30ep.py'

model = dict(
    head=dict(
        anchor_widths=(110.0, 130.0, 150.0),
        loss_bw=dict(type='MSELoss', loss_weight=20.0),
    ))

work_dir = 'work_dirs/jdm/retune/det_ideal_v1_anchor110130150_30ep'
