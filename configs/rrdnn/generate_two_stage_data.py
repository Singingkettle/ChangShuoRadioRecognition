import torch

from csrr.apis import single_gpu_test
from csrr.common.parallel import CSDataParallel
from csrr.common.utils import Config, setup_multi_processes
from csrr.datasets import build_dataloader, build_dataset
from csrr.models import build_method
from csrr.runner import (load_checkpoint, wrap_fp16_model)

versions = [
    # 1
    (4, 0.8481),
    # rician 7
    (10, 0.7818),
    (6, 0.7792),
    (7, 0.7775),
    (7, 0.7744),
    (12, 0.7726),
    (13, 0.7606),
    (11, 0.7586),
    # ray 7
    (8, 0.7796),
    (12, 0.777),
    (8, 0.7753),
    (14, 0.7711),
    (13, 0.7688),
    (12, 0.7548),
    (14, 0.7507),
    # awgn 10
    (9, 0.7781),
    (12, 0.799),
    (10, 0.8127),
    (2, 0.8185),
    (7, 0.8332),
    (3, 0.8314),
    (5, 0.8333),
    (7, 0.8302),
    (10, 0.8351),
    (6, 0.8385),

    # clockoffset 5
    (6, 0.8423),
    (5, 0.8341),
    (7, 0.8261),
    (19, 0.8166),
    (9, 0.8068),

    # real
    (13, 0.7665),

    # real different snrs
    (28, 0.6962),
    (10, 0.7207),
    (16, 0.7304),
    (21, 0.7405),
    (6, 0.7574),
    (8, 0.7617),
    (8, 0.7686),
    (8, 0.7713),
    (6, 0.7739),
    (13, 0.7753)
]


for i in range(1, 42):
    checkpoint = f'work_dirs/rrdnn_first-stage_csrr2023_v{i:d}/epoch_{versions[i-1][0]:d}.pth'
    cfg_path = f'configs/rrdnn/rrdnn_first-stage_csrr2023_v{i:d}.py'

    cfg = Config.fromfile(cfg_path)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the dataloader
    cfg.data.train['pipeline'][3]['keys'].pop(1)
    samples_per_gpu = cfg.data.train.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.train.pop('workers_per_gpu', 0)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.detector_head['cfg']['score_thr'] = 0.01
    cfg.model.detector_head['cfg']['nms'] = dict(iou_threshold=0.5)
    model = build_method(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = CSDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, cfg.dropout_alive)


    dataset.two_stage(outputs, amc_iou=0.5, mode='train_and_val')


    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 0)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)
    outputs = single_gpu_test(model, data_loader, cfg.dropout_alive)


    dataset.two_stage(outputs, amc_iou=0.5, mode='test')