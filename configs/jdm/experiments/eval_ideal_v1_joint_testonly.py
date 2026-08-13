# Fig. 13(a) ideal joint eval: merged checkpoint, v1 test split only.
_base_ = '../jdm-joint_iq-csrd.py'

test_dataloader = dict(dataset=dict(versions=['v1']))

model = dict(fuse_scores=True)

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_joint_testonly'
