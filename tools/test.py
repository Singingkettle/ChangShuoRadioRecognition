import argparse
import os

import torch

from wtisp.apis import multi_gpu_test, single_gpu_test
from wtisp.common.parallel import MMDataParallel, MMDistributedDataParallel
from wtisp.common.utils import Config, DictAction, fuse_conv_bn, mkdir_or_exist
from wtisp.dataset import build_dataloader, build_dataset
from wtisp.models import build_task
from wtisp.runner import (get_dist_info, init_dist, load_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessing test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--format-out',
        type=str,
        help='the dir to save output result file in json format')
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

    # format_out is determined in this priority: CLI > segment in file > filename
    if args.format_out is not None:
        # update configs according to CLI args if args.format_out is not None
        cfg.format_out = args.format_out
    elif cfg.get('format_out', None) is None:
        # use config filename as default format_out if cfg.format_out is None
        cfg.format_out = './format_out'

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
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=4,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_task(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
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

    rank, _ = get_dist_info()
    if rank == 0:
        if args.format_out:
            mkdir_or_exist(cfg.format_out)
            dataset.format_out(cfg.format_out, outputs)
        if hasattr(dataset, 'confusion_plot'):
            dataset.confusion_plot = False
        print(dataset.evaluate(outputs))


if __name__ == '__main__':
    main()
