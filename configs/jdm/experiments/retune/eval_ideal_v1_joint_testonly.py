# Fair ideal-protocol JOINT eval (ROOT-CAUSE FIX 2026-07-23).
#
# Same principle as eval_ideal_v1_det_testonly.py: keep the merged det+AMC
# checkpoint (trained on full simulate data) and restrict ONLY the test split
# to the ideal v1 (infdB) version. fuse_scores stays on (base default).
#
# Run with a merged joint checkpoint:
#   python tools/test_det.py <this cfg> <merged_joint_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_ideal_v1_joint_testonly
_base_ = '../../jdm-joint_iq-csrd.py'

# Restrict only the test protocol to ideal v1; joint_pipeline from base is kept
# via mmengine dict merge (we only add `versions`).
test_dataloader = dict(dataset=dict(versions=['v1']))

model = dict(fuse_scores=True)

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_joint_testonly'
