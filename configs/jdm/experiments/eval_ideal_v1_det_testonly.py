# Fig. 8(a) ideal detector eval: full-data checkpoint, v1 test split only.
_base_ = '../jdm-det_fft-csrd.py'

test_dataloader = dict(dataset=dict(versions=['v1']))

work_dir = 'work_dirs/jdm/retune/eval_ideal_v1_det_testonly'
