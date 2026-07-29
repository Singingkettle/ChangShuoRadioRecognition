# Fair ideal-protocol detector eval (ROOT-CAUSE FIX 2026-07-23).
#
# Paper "ideal" (Fig. 8a) = no random impairments / infdB == CSRD version v1.
# The correct fair comparison EVALUATES the best full-data-trained detector on
# the clean v1 test split. It must NOT retrain on v1-only: det_ideal_v1_30ep.py
# restricted train_dataloader to ['v1'] (a tiny subset) and underfit to mAP~0.31,
# which is nonsense because a noiseless test set should be EASIER than mixed.
#
# Here train/val stay on the full simulate mixture (base default versions); only
# the TEST split is restricted to v1. Run with the best full detector ckpt:
#   python tools/test_det.py <this cfg> <best_full_det_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_ideal_v1_det_testonly
_base_ = '../../jdm-det_fft-csrd.py'

# Only the test protocol becomes ideal (v1 / infdB); training data unchanged.
test_dataloader = dict(dataset=dict(versions=['v1']))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_det_testonly'
