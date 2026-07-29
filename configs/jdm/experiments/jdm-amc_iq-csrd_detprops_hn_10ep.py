# AMC fine-tune on detector proposals with hard-negative mining (10 epochs).
#
# Adds unmatched detector proposals (max IoU to any GT < 0.3) as uniform-
# target soft-label samples so the classifier learns to stay uncertain on
# leakage-dominated adjacent-band crops. Continues from the 20-ep proposal
# fine-tune best checkpoint.
_base_ = '../jdm-amc_iq-csrd.py'

proposal_cache = 'work_dirs/jdm/amc_proposals/all_splits.json'
num_classes = 5

proposal_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='PrepareGtScore', num_classes=num_classes),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

val_pipeline = [
    dict(type='LoadCSRDFrame'),
    dict(type='LoadDetProposal', proposal_cache=proposal_cache),
    dict(type='CSRDSignalToBaseband', source='frame'),
    dict(type='Reshape', shapes=dict(iq=[1, 2, 1200])),
    dict(type='PackInputs', input_key='iq'),
]

train_dataloader = dict(
    dataset=dict(
        type='CSRDModulationDetPropDataset',
        proposal_cache=proposal_cache,
        include_hard_negatives=True,
        max_hard_neg_per_frame=3,
        pipeline=proposal_pipeline,
    ),
)
val_dataloader = dict(
    dataset=dict(
        type='CSRDModulationDetPropDataset',
        proposal_cache=proposal_cache,
        include_hard_negatives=False,
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    dataset=dict(
        type='CSRDModulationDetPropDataset',
        proposal_cache=proposal_cache,
        include_hard_negatives=False,
        pipeline=val_pipeline,
    ),
)

model = dict(
    head=dict(
        loss=dict(
            type='CrossEntropyLoss',
            use_soft=True,
            loss_weight=1.0,
        ),
    ),
)

load_from = (
    'work_dirs/jdm/exp_amc_detprops_20ep/best_accuracy_top1_epoch_20.pth')

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=5e-5),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=10,
    eta_min=1e-6,
)

train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
work_dir = 'work_dirs/jdm/exp_amc_detprops_hn_10ep'
