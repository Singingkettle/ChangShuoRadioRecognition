# Paper-exact Fig. 8(a) SIMULATE detector eval (tightened 2026-07-24).
#
# Paper Sec. VI-B (Fig. 8): "simulated setting" = realistic environment with
# various interference (channel / velocity / K-factor) and SNR as a random
# factor — NOT the full 124-version mixture (which also mixes in ideal, pure
# AWGN, and single-factor ablation versions and inflates the simulate bar).
#
# Maps to twc/generate.m blocks:
#   Real       -> v104
#   Real_awgn  -> v105..v124  (random fading+clock, fixed SNR per version)
#
# Train/val stay on the full mixture (base default); ONLY the TEST split is
# restricted. Run with a full-data-trained detector ckpt:
#   python tools/test_det.py <this cfg> <best_full_det_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_simulate_real_awgn_det_testonly
_base_ = '../../jdm-det_fft-csrd.py'

# v104 + real_awgn SNR ladder (paper simulate aggregate).
_simulate_versions = ['v104'] + [f'v{i}' for i in range(105, 125)]
test_dataloader = dict(dataset=dict(versions=_simulate_versions))

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_det_testonly'
