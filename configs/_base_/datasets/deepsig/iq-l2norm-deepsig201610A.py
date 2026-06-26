# RML2016.10A IQ pipeline with per-sample L2 (unit-energy) normalization.
#
# RML2016.10A ships each example at a tiny fixed scale (Frobenius norm ~0.1,
# RMS ~0.006). Deep recurrent models (MCLDNN, CLDNN, CGDNet) sit in the
# near-linear gate regime at that scale and converge to a markedly worse
# optimum than the AMR-Benchmark reference. SelfNormalize divides each sample
# by its Frobenius norm (-> unit energy, ~10x scale-up). On MCLDNN this lifts
# peak accuracy from 85% to 93% and overall from 58% to 62% (>= reference),
# while pure-CNN models are marginally hurt by it -- hence this normalized
# variant is used only by the recurrent IQ models, and the plain
# ``iq-deepsig201610A.py`` (no normalization) is kept for the CNNs.
_base_ = ['./iq-deepsig201610A.py']

pipeline = [
    dict(type='SelfNormalize', norms=dict(iq={})),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 128])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(dataset=dict(pipeline=pipeline))
val_dataloader = dict(dataset=dict(pipeline=pipeline))
test_dataloader = dict(dataset=dict(pipeline=pipeline))
