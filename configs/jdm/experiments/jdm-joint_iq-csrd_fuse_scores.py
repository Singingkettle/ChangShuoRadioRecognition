# Joint JDM evaluation with detection x classification score fusion.
_base_ = '../jdm-joint_iq-csrd.py'

model = dict(fuse_scores=True)
work_dir = 'work_dirs/jdm/jdm-joint_iq-csrd_detprops_20ep_fuse'
