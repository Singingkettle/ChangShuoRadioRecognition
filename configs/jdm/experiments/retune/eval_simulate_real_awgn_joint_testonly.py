# Paper-exact Fig. 13(a) SIMULATE joint eval (tightened 2026-07-24).
#
# Same protocol as eval_simulate_real_awgn_det_testonly.py: restrict ONLY the
# TEST split to generate.m Real (v104) + Real_awgn (v105..v124). Merged
# det+AMC checkpoint stays full-data-trained; fuse_scores on.
#
#   python tools/test_det.py <this cfg> <merged_joint_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_simulate_real_awgn_joint_testonly
_base_ = '../../jdm-joint_iq-csrd.py'

_simulate_versions = ['v104'] + [f'v{i}' for i in range(105, 125)]
test_dataloader = dict(dataset=dict(versions=_simulate_versions))

model = dict(fuse_scores=True)

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_joint_testonly'
