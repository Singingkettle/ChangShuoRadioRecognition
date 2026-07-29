# Diagnostic: simulate (Real + Real_awgn) JOINT eval WITH per-IoU AP breakdown.
#
# Same protocol as eval_simulate_real_awgn_joint_testonly.py; class-aware
# evaluator also emits per-IoU AP to decompose the joint mAP gap.
#   python tools/test_det.py <this cfg> <merged_joint_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_simulate_real_awgn_joint_periou
_base_ = '../../jdm-joint_iq-csrd.py'

_simulate_versions = ['v104'] + [f'v{i}' for i in range(105, 125)]
test_dataloader = dict(dataset=dict(versions=_simulate_versions))

model = dict(fuse_scores=True)

test_evaluator = dict(
    type='SignalDetectionMetric', classwise=True, per_iou_ap=True)

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_joint_periou'
