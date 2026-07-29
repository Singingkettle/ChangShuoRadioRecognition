# Paper Fig. 7/10/12 AWGN SNR-subset detector eval (NOT Fig. 8 simulate).
#
# Paper Table I SNR grid [12:30:2] maps to ChangShuoTwc2026 AWGN versions
# v89..v98 (generate.m Awgn block; velocity=0, no fading). Use for SNR-curve
# comparison only — do not score this against Fig. 8/13 simulate bars.
_base_ = '../../jdm-det_fft-csrd.py'

test_dataloader = dict(
    dataset=dict(
        versions=[
            'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98'
        ]))

work_dir = 'work_dirs/jdm/retune/eval_awgn_v89_v98_det_testonly'
