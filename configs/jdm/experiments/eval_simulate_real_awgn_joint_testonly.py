# Fig. 13(a) simulate joint eval: Real (v104) + Real_awgn (v105–v124).
_base_ = '../jdm-joint_iq-csrd.py'

_simulate_versions = ['v104'] + [f'v{i}' for i in range(105, 125)]
test_dataloader = dict(dataset=dict(versions=_simulate_versions))

model = dict(fuse_scores=True)

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_joint_testonly'
