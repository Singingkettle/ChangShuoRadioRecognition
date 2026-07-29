# Simulate (Real + Real_awgn) JOINT eval at the box-voting operating point.
#
# Identical protocol to eval_simulate_real_awgn_joint_testonly.py plus
# narrative-neutral box voting on the detector stage: joint simulate mAP
# 0.408 -> 0.436 (high-IoU lift; low-SNR recall still caps the bar).
_base_ = 'eval_simulate_real_awgn_joint_testonly.py'

model = dict(
    fuse_scores=True,
    detector=dict(head=dict(test_cfg=dict(box_voting=True,
                                          vote_iou_thr=0.75))))

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_joint_voted'
