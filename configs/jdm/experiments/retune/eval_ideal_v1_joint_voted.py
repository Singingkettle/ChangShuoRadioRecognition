# Ideal (v1) JOINT eval at the box-voting operating point (2026-07-29).
#
# Identical protocol to eval_ideal_v1_joint_testonly.py plus narrative-neutral
# box voting on the detector stage: joint ideal mAP 0.708 -> 0.762. Score
# fusion stays at the exact-paper alpha=1 / T=1 (calibration is rank-preserving
# and does not move class-aware mAP).
_base_ = 'eval_ideal_v1_joint_testonly.py'

model = dict(
    fuse_scores=True,
    detector=dict(head=dict(test_cfg=dict(
        box_voting=True, vote_iou_thr=0.75, vote_score_pow=4.5))))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_joint_voted'
