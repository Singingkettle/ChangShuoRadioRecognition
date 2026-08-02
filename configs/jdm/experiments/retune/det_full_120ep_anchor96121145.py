"""Wave-30: k-means refined anchors 96/121/145 — same det120 schedule as champion.

Train-set GT bandwidths cluster at 96.5/120.6/144.7 (sps 15/12/10). Current
96/120/146 already near-optimal; this is the 1-bin refinement from k-means.
"""
_base_ = './det_full_120ep_lr1e3.py'

model = dict(head=dict(anchor_widths=(96.0, 121.0, 145.0)))

work_dir = 'work_dirs/jdm/retune/det_full_120ep_anchor96121145'
