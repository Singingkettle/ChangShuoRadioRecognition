import argparse
import os
import os.path as osp

import torch

from wtisp.common.utils import Config, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessing Generate Train.sh File')
    parser.add_argument('config', help='plot config file path')
    parser.add_argument('--scripts_dir', help='dir to save the train.sh files')
    parser.add_argument('--group_num', default=8, type=int, help='number of configs in one train.sh file')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--multi_gpu', default=True, type=bool, help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    configs = set()

    if args.scripts_dir is not None:
        scripts_dir = osp.join(args.scripts_dir, 'train_' + cfg.plot['config'])
    else:
        scripts_dir = osp.join('./scripts', 'train_' + cfg.plot['config'])

    # create work_dir
    mkdir_or_exist(osp.abspath(scripts_dir))

    # Add config from confusion_maps
    if 'confusion_maps' in cfg.plot:
        if isinstance(cfg.plot['confusion_maps'], dict):
            config = cfg.plot['confusion_maps']['config']
            if config not in configs:
                configs.add(config)

        elif isinstance(cfg.plot['confusion_maps'], list):
            for confusion_map in cfg.plot['confusion_maps']:
                config = confusion_map['config']
                if config not in configs:
                    configs.add(config)
        else:
            raise ValueError('The confusion maps must be list or dict!')

    # Add config from train_test_curves
    if 'train_test_curves' in cfg.plot:
        train_test_curves = cfg.plot['train_test_curves']
        if isinstance(train_test_curves, dict):
            for method in train_test_curves['methods']:
                if method['config'] not in configs:
                    configs.add(method['config'])
        elif isinstance(train_test_curves, list):
            for train_test_curve in train_test_curves:
                for method in train_test_curve['methods']:
                    if method['config'] not in configs:
                        configs.add(method['config'])
        else:
            raise ValueError('The train test curves must be list or dict!')

    # Add config from snr_modulation
    if 'snr_modulation' in cfg.plot:
        snr_modulation = cfg.plot['snr_modulation']
        if isinstance(snr_modulation, dict):
            for method in snr_modulation['methods']:
                if method['config'] not in configs:
                    configs.add(method['config'])
        elif isinstance(snr_modulation, list):
            for snr_accuracy in snr_modulation:
                for method in snr_accuracy['methods']:
                    if method['config'] not in configs:
                        configs.add(method['config'])
        else:
            raise ValueError('The snr_modulation must be list or dict!')

    base_master_port = 29647

    method_index = 0
    train_sh = ''
    count_index = 0
    group_index = 0
    gpu_num = torch.cuda.device_count()
    configs = list(configs)
    configs = sorted(configs)

    if not args.multi_gpu:
        args.group_num = gpu_num

    for config in configs:
        if args.multi_gpu:
            python_sh = 'nohup python -m torch.distributed.launch --nproc_per_node={} --master_port={} tools/train.py'.format(
                gpu_num, base_master_port + method_index)
        else:
            python_sh = 'nohup python tools/train.py'
        config_sh = ' ./configs/{}/{}'.format(config.split('_')[0], config + '.py')

        work_dir_sh = ' --work_dir {}'.format(args.work_dir)
        if args.multi_gpu:
            end_sh = ' --seed 0 --launcher pytorch > /dev/null 2>&1 &\n\n\n\n'
        else:
            end_sh = ' --gpu-ids {:d} --seed 0 > /dev/null 2>&1 &\n\n\n\n'.format(count_index)

        start_info = 'echo \"Start Train: {}\"\n\n'.format(config_sh)
        train_sh = train_sh + start_info + python_sh + config_sh + work_dir_sh + end_sh
        method_index += 1
        count_index += 1
        if count_index >= args.group_num:
            with open(os.path.join(scripts_dir, 'train_{:02d}.sh'.format(group_index)), 'w') as f:
                f.write(train_sh)
            train_sh = ''
            count_index = 0
            group_index += 1

    if train_sh is not '':
        with open(os.path.join(scripts_dir, 'train_{:02d}.sh'.format(group_index)), 'w') as f:
            f.write(train_sh)


if __name__ == '__main__':
    main()
