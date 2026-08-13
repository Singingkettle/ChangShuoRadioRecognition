# Fig. 7/10/12 AWGN SNR-subset eval (v89–v98). Not Fig. 8/13 simulate.
_base_ = '../jdm-det_fft-csrd.py'

test_dataloader = dict(
    dataset=dict(
        versions=[
            'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98'
        ]))

work_dir = 'work_dirs/jdm/retune/eval_awgn_v89_v98_det_testonly'
