# Diagnostic: ideal (v1) JOINT eval WITH per-IoU class-aware AP breakdown.
#
# Same protocol as eval_ideal_v1_joint_testonly.py (merged det+AMC ckpt, test
# restricted to v1, fuse_scores on). The class-aware evaluator also emits AP at
# each IoU threshold so we can tell whether the joint mAP deficit is dominated
# by detection localization (high-IoU AP) or classification (all-IoU offset).
#   python tools/test_det.py <this cfg> <merged_joint_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_ideal_v1_joint_periou
_base_ = '../../jdm-joint_iq-csrd.py'

test_dataloader = dict(dataset=dict(versions=['v1']))

model = dict(fuse_scores=True)

test_evaluator = dict(
    type='SignalDetectionMetric', classwise=True, per_iou_ap=True)

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_joint_periou'
