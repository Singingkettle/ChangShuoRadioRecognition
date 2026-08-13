# Ideal joint eval at the box-voting operating point.
_base_ = './eval_ideal_v1_joint_testonly.py'

model = dict(
    fuse_scores=True,
    detector=dict(head=dict(test_cfg=dict(
        box_voting=True, vote_iou_thr=0.75, vote_score_pow=4.5))))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_joint_voted'
