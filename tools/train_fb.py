import argparse
import os
import os.path as osp
import time

import numpy as np

from wtisp.common import get_root_logger, collect_env
from wtisp.common.utils import Config, mkdir_or_exist
from wtisp.datasets import build_dataset
from wtisp.models import build_fb


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessingTrain a regression model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join(
            args.work_dir, osp.splitext(osp.basename(args.config))[0])
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
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

    meta['exp_name'] = osp.basename(args.config)

    cfg.model['model_path'] = os.path.join(cfg.work_dir, 'model.fb')
    model = build_fb(cfg.model)

    dataset = build_dataset(cfg.data.train)
    if hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    data = []
    label = []
    for idx in range(len(dataset)):
        data.append(dataset[idx][cfg.x])
        label.append(dataset[idx][cfg.y])

    data = np.concatenate(data)
    label = np.array(label)

    # Train
    model(data, label)


if __name__ == '__main__':
    main()
