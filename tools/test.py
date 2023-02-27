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
from csrr.runner import (get_dist_info, init_dist, load_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition test (and eval) a model')
    parser.add_argument('figure_configs', help='test figure_configs file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        help='the directory to format the file containing evaluation figures, results in pickle format, '
             'and format data to upload into test server')
    parser.add_argument('--out', action='store_true', help='output result file in pickle format')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        action='store_true',
        help='evaluation on the dataset')
    parser.add_argument(
        '--fuse_conv_bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used figure_configs, the key-value pair '
             'in xxx=yyy format will be merged into figure_configs file.')
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

    assert args.out or args.eval or args.format_only, \
        ('Please specify at least one operation (format/eval/format the '
         'results / format the results) with the argument "--out", "--eval"'
         ', "--format-only"')

    cfg = Config.fromfile(args.config)

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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 4)
    workers_per_gpu = cfg.data.test.pop('workers_per_gpu', 4)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_method(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = CSDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, cfg.dropout_alive)
    else:
        model = CSDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, cfg.dropout_alive)

    rank, _ = get_dist_info()
    # allows not to create
    mkdir_or_exist(os.path.abspath(args.work_dir))

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.work_dir}')
            IODump(outputs, os.path.join(args.work_dir, 'results.pkl'))
        if args.format_only:
            dataset.format_results(args.work_dir, outputs)
        if args.eval:
            metric = dataset.evaluate(outputs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                json_file = os.path.join(args.work_dir, f'eval_{timestamp}.json')
                IODump(metric_dict, json_file)


if __name__ == '__main__':
    main()
