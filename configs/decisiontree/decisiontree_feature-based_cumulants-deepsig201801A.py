log_level = 'INFO'

_base_ = [
    '../_base_/datasets/deepsig/cumulants-deepsig201801A.py',
]

model = dict(
    type='DecisionTree',
    max_depth=6,

)
