import pandas as pd
import torch

from csrr.apis import single_gpu_test
from csrr.common.fileio import load as IOLoad
from csrr.common.parallel import CSDataParallel
from csrr.common.utils import Config, setup_multi_processes
from csrr.datasets import build_dataloader, build_dataset
from csrr.models import build_method
from csrr.runner import (load_checkpoint, wrap_fp16_model)


def get_amc(i):
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

    checkpoint = f'work_dirs/rrdnn_second-stage_csrr2023_v{i:d}/epoch_{versions[i-1][0]:d}.pth'
    cfg_path = f'configs/rrdnn/rrdnn_second-stage_csrr2023_v{i:d}.py'
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

    return results


def get_det(i):
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
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 0)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
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

    return outputs


channels = []
cols = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "none"]
all_data = []
keys = None
for i in range(1, 42):

    det_results = get_det(i)
    amc_results = get_amc(i)
    data_json = f'data/ChangShuo/v{i:d}/validation.json'
    info_dict = IOLoad(data_json)
    channels.append(info_dict['channels'][0])

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
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 0)
    cfg.data.test['is_only_det'] = False
    dataset = build_dataset(cfg.data.test)
    eval_results = dataset.merge(det_results, amc_results)
    keys = sorted(eval_results.keys())
    res = []
    for key in keys:
        res.append(eval_results[key])

    all_data.append(res)

df1 = pd.DataFrame(all_data, index=channels, columns=keys)
df1.to_excel("final.xlsx")



# channels = []
# cols = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "none"]
# all_data = []
# keys = None
#
# i = 55
# det_results = get_det(i)
# amc_results = get_amc(i)
#
# cfg_path = f'configs/rrdnn/rrdnn_first-stage_csrr2023_v{i:d}.py'
# cfg = Config.fromfile(cfg_path)
# # set multi-process settings
# setup_multi_processes(cfg)
#
# # set cudnn_benchmark
# if cfg.get('cudnn_benchmark', False):
#     torch.backends.cudnn.benchmark = True
#
# # in case the test dataset is concatenated
# if isinstance(cfg.data.test, dict):
#     cfg.data.test.test_mode = True
# elif isinstance(cfg.data.test, list):
#     for ds_cfg in cfg.data.test:
#         ds_cfg.test_mode = True
#
# # build the dataloader
# samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
# workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 0)
# cfg.data.test['is_only_det'] = False
# dataset = build_dataset(cfg.data.test)
#
# snr_to_imgids = dict()
#
#
# for id, anno in enumerate(dataset.data_infos['annotations']):
#     if anno['snr'][0] in snr_to_imgids:
#             snr_to_imgids[anno['snr'][0]].append(id)
#     else:
#         snr_to_imgids[anno['snr'][0]] = []
#
#
# data, rows, cols = dataset.merge_v55(det_results, amc_results, snr_to_imgids)
# df1 = pd.DataFrame(data, index=rows, columns=cols)
# df1.to_excel("final_v55.xlsx")