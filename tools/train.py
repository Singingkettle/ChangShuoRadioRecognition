import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
import torch.distributed as dist

from csrr import __version__
from csrr.apis import init_random_seed, set_random_seed, train_task
from csrr.common import get_root_logger, collect_env
from csrr.common.utils import DictAction, Config, mkdir_or_exist, redir_and_exist
from csrr.datasets import build_dataset
from csrr.models import build_task
from csrr.runner import init_dist, get_dist_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognitionTrain a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs, models and results')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--no_validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu_ids', type=int, nargs='+',
                            help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff-seed', action='store_true',
                        help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file (deprecate), '
                             'change to --cfg-options instead.')
    parser.add_argument('--cfg_options', nargs='+', action=DictAction,
                        help='override some settings in the used config, '
                             'the key-value pair in xxx=yyy format will be merged into config file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg_options cannot be both '
            'specified, --options is deprecated in favor of --cfg_options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg_options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    # load cfg
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join(
            args.work_dir, osp.splitext(osp.basename(args.config))[0])
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        rank = 0
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))

    # If not reuse from history checkpoint and there
    # are history checkpoint files, then remove the whole work_dir
    if rank == 0:
        if cfg.resume_from is None and not cfg.auto_resume:
            redir_and_exist(cfg.work_dir)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_task(cfg.model)

    # log network structure
    logger.info(f'Network Structure:\n{model}')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(csrr_version=__version__)

    if hasattr(datasets[0], 'CLASSES'):
        model.CLASSES = datasets[0].CLASSES
        meta['CLASSES'] = datasets[0].CLASSES

    train_task(model, datasets, cfg, distributed=distributed, validate=(not args.no_validate),
               timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    main()
