# Ideal (v1) detector eval at the box-voting operating point.
#
# Identical protocol to eval_ideal_v1_det_testonly.py, plus inference-time
# box voting. vote_score_pow=4.5 is the promoted OP (ideal det mAP 0.8254).
# Exact-paper NMS stays in eval_ideal_v1_det_testonly.py.
_base_ = 'eval_ideal_v1_det_testonly.py'

model = dict(head=dict(test_cfg=dict(
    box_voting=True, vote_iou_thr=0.75, vote_score_pow=4.5)))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_det_voted'
