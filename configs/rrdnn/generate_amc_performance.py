import torch

from csrr.apis import single_gpu_test
from csrr.common.parallel import CSDataParallel
from csrr.common.utils import Config, setup_multi_processes
from csrr.datasets import build_dataloader, build_dataset
from csrr.models import build_method
from csrr.runner import (load_checkpoint, wrap_fp16_model)
from csrr.common.utils.config import load_json_log
from csrr.common.fileio import load as IOLoad
import pandas as pd

channels = []
cols = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "none"]
all_data_acc = []
all_data_f = []
versions = [
    # ideal
    (99, 0.9207),

    # rician speed
    (100, 0.6058),
    (50, 0.5222),
    (77, 0.5104),
    (30, 0.5073),
    (33, 0.4937),
    (30, 0.4829),
    (33, 0.4678),

    # ray speed
    (59, 0.5957),
    (74, 0.5112),
    (75, 0.5047),
    (18, 0.4943),
    (19, 0.4801),
    (22, 0.4709),
    (29, 0.4552),

    # awgn snr
    (85, 0.7996),
    (97, 0.8483),
    (95, 0.8516),
    (85, 0.8595),
    (51, 0.8757),
    (64, 0.8711),
    (88, 0.878),
    (44, 0.8892),
    (39, 0.8913),
    (83, 0.8851),

    # clock offset
    (95, 0.8603),
    (93, 0.854),
    (36, 0.8454),
    (48, 0.8382),
    (83, 0.8316),

    # real
    (98, 0.6135),

    # real snr
    (96, 0.5302),
    (95, 0.556),
    (18, 0.4763),
    (93, 0.5821),
    (71, 0.5303),
    (81, 0.5837),
    (71, 0.632),
    (70, 0.6358),
    (30, 0.5455),
    (68, 0.6646),
]
for i in range(1, 42):
    checkpoint = f'work_dirs/rrdnn_second-stage_csrr2023_v{i:d}/epoch_{versions[i-1][0]:d}.pth'
    cfg_path = f'configs/rrdnn/rrdnn_second-stage_csrr2023_v{i:d}.py'
    data_json = f'data/ChangShuo/v{i:d}/validation.json'
    cfg = Config.fromfile(cfg_path)
    info_dict = IOLoad(data_json)
    channels.append(info_dict['channels'][0])

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

    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 0)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        is_det=False,
        shuffle=False)

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
    results = single_gpu_test(model, data_loader, cfg.dropout_alive)
    performance = dataset.rrdnn(results, cfg)

    cur_accuracy = dict()
    accuracy = performance.ACC
    F1Score = performance.F1

    accs = [accuracy]

    f1s = []
    for i in range(len(cols)):
        f1s.append(F1Score[cols[i]])

    all_data_acc.append(accs)
    all_data_f.append(f1s)

df1 = pd.DataFrame(all_data_acc, index=channels, columns=['acc'])
df1.to_excel("amc_acc.xlsx")

df2 = pd.DataFrame(all_data_f, index=channels, columns=cols)
df2.to_excel("amc_f1.xlsx")