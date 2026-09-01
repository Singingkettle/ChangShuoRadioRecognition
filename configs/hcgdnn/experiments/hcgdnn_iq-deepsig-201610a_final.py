# HCGDNN final training on the merged RadioML.2016.10A 60% split.
# Paper: "A Hierarchical Classification Head based Convolutional Gated Deep Neural Network for Automatic Modulation Classification", IEEE Transactions on Wireless Communications (2022).

_base_ = ['../hcgdnn_iq-deepsig-201610a.py']

train_dataloader = dict(
    dataset=dict(ann_file='train_and_validation.json'))
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = dict(
    _delete_=True,
    batch_size=640,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AMCDataset',
        data_root=('data/ModulationClassification/DeepSig/'
                   'RadioML.2016.10A'),
        ann_file='test.json',
        pipeline={{_base_.pipeline}},
        cache=True,
        test_mode=True))
custom_hooks = []
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=10,
        save_last=True,
        max_keep_ckpts=3))
