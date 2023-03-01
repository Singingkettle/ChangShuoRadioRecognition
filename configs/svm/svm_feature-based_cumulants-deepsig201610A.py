log_level = 'INFO'

_base_ = [
    '../_base_/datasets/deepsig/cumulants-deepsig201610A.py',
]

model = dict(
    type='SVM',
    regularization=1,
    max_iter=50000,

)
