import argparse
import os
import os.path as osp

import torch

from wtisp.apis import multi_gpu_test, single_gpu_test
from wtisp.common.parallel import MMDataParallel, MMDistributedDataParallel
from wtisp.common.utils import Config, DictAction, fuse_conv_bn, mkdir_or_exist, get_the_best_checkpoint
from wtisp.dataset import build_dataloader, build_dataset
from wtisp.models import build_task
from wtisp.runner import (get_dist_info, init_dist, load_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessing get distillation knowledge of a model')
    parser.add_argument('config', help='config file path to provide distillation knowledge')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg.pretty_text)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = './work_dirs'

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 4)

    cfg.data.test['ann_file'] = 'train_and_val.json'
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    config = osp.splitext(osp.basename(args.config))[0]
    best_epoch = get_the_best_checkpoint(cfg.work_dir, config)
    checkpoint = cfg.work_dir + '/{}/epoch_{}.pth'.format(config, best_epoch)
    model = build_task(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, cfg.dropout_alive)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, cfg.dropout_alive)

    format_out_distillation = osp.join(cfg.work_dir, config, 'format_out_distillation')
    rank, _ = get_dist_info()
    if rank == 0:
        mkdir_or_exist(format_out_distillation)
        dataset.format_out(format_out_distillation, outputs)


if __name__ == '__main__':
    main()
