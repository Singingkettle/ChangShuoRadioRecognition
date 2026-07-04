# Detector-only inference sensitivity config for NMS suppression.
#
# The baseline often produces saturated objectness scores for multiple anchors
# at the same center. A very loose NMS can recover suppressed wide boxes but
# full max_per_frame=20 evaluation is slow, so this keeps only the top 6
# proposals per frame as a bounded probe.
_base_ = '../jdm-det_fft-csrd.py'

model = dict(
    head=dict(
        anchor_widths=(96.0, 120.0, 146.0),
        test_cfg=dict(score_thr=0.05, nms_iou_thr=0.85, max_per_frame=6),
    ))

work_dir = 'work_dirs/jdm/exp_det_anchor096146_nms085_top6'
