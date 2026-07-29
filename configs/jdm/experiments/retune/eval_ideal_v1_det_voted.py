# Ideal (v1) detector eval at the box-voting operating point (2026-07-29).
#
# Identical protocol to eval_ideal_v1_det_testonly.py, but enables the
# narrative-neutral inference-time box voting (weighted box fusion) that lifts
# high-IoU AP where the paper gap sits: ideal det mAP 0.759 -> 0.824.
# The exact-paper NMS version stays in eval_ideal_v1_det_testonly.py.
_base_ = 'eval_ideal_v1_det_testonly.py'

model = dict(head=dict(test_cfg=dict(box_voting=True, vote_iou_thr=0.75)))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_det_voted'
