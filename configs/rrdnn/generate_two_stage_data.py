import argparse
import os
import time

import torch

from csrr.apis import multi_gpu_test, single_gpu_test
from csrr.common.fileio import dump as IODump
from csrr.common.parallel import CSDataParallel, CSDistributedDataParallel
from csrr.common.utils import Config, DictAction, fuse_conv_bn, mkdir_or_exist, setup_multi_processes
from csrr.datasets import build_dataloader, build_dataset
from csrr.models import build_method
from csrr.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)

checkpoint = 'work_dirs/rrdnn_first-stage_csrr2023/googlenet.pth'
cfg_path = 'configs/rrdnn/rrdnn_first-stage_csrr2023.py'

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


dataset.two_stage(outputs, amc_iou=0.5)