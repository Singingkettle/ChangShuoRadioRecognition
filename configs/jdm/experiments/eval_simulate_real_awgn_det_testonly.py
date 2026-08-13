# Fig. 8(a) simulate detector eval: Real (v104) + Real_awgn (v105–v124).
_base_ = '../jdm-det_fft-csrd.py'

_simulate_versions = ['v104'] + [f'v{i}' for i in range(105, 125)]
test_dataloader = dict(dataset=dict(versions=_simulate_versions))

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_det_testonly'
