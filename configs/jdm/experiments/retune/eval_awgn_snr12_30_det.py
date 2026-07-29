# Fair-comparison detector test: AWGN Table I SNR subset [12:30:2] (v89–v98).
_base_ = '../../jdm-det_fft-csrd.py'

_paper_snr_versions = [
    'v89', 'v90', 'v91', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98'
]

train_dataloader = dict(dataset=dict(versions=_paper_snr_versions))
val_dataloader = dict(dataset=dict(versions=_paper_snr_versions))
test_dataloader = dict(dataset=dict(versions=_paper_snr_versions))

work_dir = 'work_dirs/jdm/retune/eval_awgn_snr12_30_det'
