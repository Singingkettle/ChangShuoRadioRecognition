# Fair ideal-protocol joint eval: CSRD v1 only (no random impairments / infdB).
_base_ = '../../jdm-joint_iq-csrd.py'

_ideal_versions = ['v1']

val_dataloader = dict(dataset=dict(versions=_ideal_versions))
test_dataloader = dict(dataset=dict(versions=_ideal_versions))

model = dict(fuse_scores=True)

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_joint'
