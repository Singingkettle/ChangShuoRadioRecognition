# Simulate (Real + Real_awgn) detector eval at the box-voting operating point.
#
# Identical protocol to eval_simulate_real_awgn_det_testonly.py plus
# narrative-neutral box voting (vote_iou_thr=0.75). Simulate gains are smaller
# than ideal because the simulate bar is dominated by low-SNR recall (AP50
# ~0.97 -> mAP limited), which voting cannot recover; it still nets a positive
# high-IoU lift.
_base_ = 'eval_simulate_real_awgn_det_testonly.py'

model = dict(head=dict(test_cfg=dict(box_voting=True, vote_iou_thr=0.75)))

work_dir = 'work_dirs/jdm/retune/eval_simulate_real_awgn_det_voted'
