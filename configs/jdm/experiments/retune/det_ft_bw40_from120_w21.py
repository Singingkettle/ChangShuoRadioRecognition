# Wave-21 detector fine-tune: tighten boxes to attack the high-IoU tail.
#
# A1 finding: det AP is ~1.0 through IoU 0.80 then cliffs (AP90=0.20, AP95=0.07);
# box voting recovers AP85 but not AP95. Boxes are a few bins loose. This
# continues from the best detector (det_full_120ep ep4) and DOUBLES the
# bandwidth-regression loss weight (20 -> 40) at low LR so the width head is
# penalized harder for loose intervals. Architecture unchanged (in freeze).
_base_ = '../../jdm-det_fft-csrd.py'

load_from = 'work_dirs/jdm/retune/det_full_120ep_lr1e3/best_detection_mAP_epoch_4.pth'

model = dict(head=dict(loss_bw=dict(type='MSELoss', loss_weight=40.0)))

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=True, T_max=40, eta_min=1e-6)
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

work_dir = 'work_dirs/jdm/retune/det_ft_bw40_from120_w21'
