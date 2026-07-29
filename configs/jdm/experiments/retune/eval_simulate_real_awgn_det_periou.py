# Diagnostic: simulate (Real + Real_awgn) detector eval WITH per-IoU AP break.
#
# Same protocol as eval_simulate_real_awgn_det_testonly.py; the evaluator also
# emits AP at each IoU threshold to localize where the simulate mAP gap sits.
#   python tools/test_det.py <this cfg> <best_full_det_ckpt.pth> \
#       --work-dir work_dirs/jdm/retune/eval_simulate_real_awgn_det_periou
_base_ = '../../jdm-det_fft-csrd.py'

_simulate_versions = ['v104'] + [f'v{i}' for i in range(105, 125)]
test_dataloader = dict(dataset=dict(versions=_simulate_versions))

test_evaluator = dict(type='SignalDetectionMetric', per_iou_ap=True)

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_det_periou'
