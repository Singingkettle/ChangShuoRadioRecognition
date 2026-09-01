# MLDNN final training on the merged RadioML.2018.01A 60% split.
# Paper: "Multitask-Learning-Based Deep Neural Network for Automatic Modulation Classification", IEEE Internet of Things Journal (2021).

_base_ = ['../mldnn_iq-ap-deepsig-201801a.py']

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
                   'RadioML.2018.01A'),
        ann_file='test.json',
        pipeline={{_base_.test_pipeline}},
        cache=True,
        cache_file='auto',
        test_mode=True))
train_cfg = dict(max_epochs=370)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=1))
