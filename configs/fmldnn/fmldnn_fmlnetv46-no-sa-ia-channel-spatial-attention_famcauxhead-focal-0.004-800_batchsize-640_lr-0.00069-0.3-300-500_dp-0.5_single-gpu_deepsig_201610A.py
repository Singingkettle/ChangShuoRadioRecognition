checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', add_graph=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/home/citybuster/Data/SignalProcessing/Workdir/fmldnn_fmlnetv46-no-sa-ia-channel-spatial-attention_famcauxhead-focal-0.004-800_batchsize-640_lr-0.00069-0.3-300-500_dp-0.5_single-gpu_deepsig_201610A/epoch_600.pth'
workflow = [('train', 1)]
evaluation = dict(interval=10)
dropout_alive = False
batch_size = 640
channel_mode = True
dataset_type = 'WTIMCDataset'
data_root = '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A'
data = dict(
    samples_per_gpu=640,
    workers_per_gpu=2,
    train=dict(
        type='WTIMCDataset',
        data_root=
        '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A',
        ann_file='train_and_val.json',
        iq=True,
        ap=True,
        channel_mode=True,
        use_cache=True),
    val=dict(
        type='WTIMCDataset',
        data_root=
        '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A',
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=True,
        use_cache=True),
    test=dict(
        type='WTIMCDataset',
        data_root=
        '/home/citybuster/Data/SignalProcessing/ModulationClassification/DeepSig/201610A',
        ann_file='test.json',
        iq=True,
        ap=True,
        channel_mode=True))
in_features = 100
out_features = 256
num_classes = 11
model = dict(
    type='FMLDNN',
    channel_mode=True,
    backbone=dict(
        type='FMLNetV46',
        in_features=4,
        channel_mode=True,
        skip_connection=True),
    classifier_head=dict(
        type='FAMCAUXHead',
        in_features=100,
        out_features=256,
        num_classes=11,
        batch_size=640,
        loss_cls=dict(type='FocalLoss', loss_weight=1, alpha=0.5),
        aux_head=dict(
            type='IntraOrthogonalHead',
            in_features=256,
            batch_size=640,
            num_classes=11,
            mm='inner_product',
            is_abs=False,
            loss_aux=dict(
                type='LogisticLoss', loss_weight=0.004, temperature=0.00125))))
train_cfg = dict()
test_cfg = dict(vis_fea=False)
total_epochs = 1200
optimizer = dict(type='Adam', lr=0.00069)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', gamma=0.3, step=[300, 500, 1000])
work_dir = '/home/citybuster/Data/SignalProcessing/Workdir/fmldnn_fmlnetv46-no-sa-ia-channel-spatial-attention_famcauxhead-focal-0.004-800_batchsize-640_lr-0.00069-0.3-300-500_dp-0.5_single-gpu_deepsig_201610A'
gpu_ids = [7]
