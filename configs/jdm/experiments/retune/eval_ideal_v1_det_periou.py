# Diagnostic: ideal (v1) detector eval WITH per-IoU AP breakdown.
#
# Same protocol as eval_ideal_v1_det_testonly.py (train/val on full mixture,
# test restricted to v1) but the evaluator also emits AP at each IoU threshold
# (AP_iou_0.50 .. AP_iou_0.95). Goal: localize the paper-radar gap
# (Fig. 8a ideal mAP 0.91 vs our ~0.80). AP50/AP75 already meet/exceed paper,
# so the deficit is expected to sit in the high-IoU (box-tightness) regime.
#   python tools/test_det.py <this cfg> <best_full_det_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_ideal_v1_det_periou
_base_ = '../../jdm-det_fft-csrd.py'

test_dataloader = dict(dataset=dict(versions=['v1']))

test_evaluator = dict(type='SignalDetectionMetric', per_iou_ap=True)

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_det_periou'
