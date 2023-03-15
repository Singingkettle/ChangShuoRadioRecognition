import argparse
import os
import os.path as osp

import torch

from csrr.common.utils import Config, mkdir_or_exist, filter_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition Generate Train.sh File')
    parser.add_argument('config', help='performance config file path')
    parser.add_argument('--scripts_dir', help='dir to format the train.sh files')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='run multi python scripts')
    parser.add_argument('--group_num', default=3, type=int, help='number of train.sh file')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use parallel gpu training')
    parser.add_argument('--is_regeneration', default=False, action='store_true',
                        help='use parallel training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    if args.scripts_dir is not None:
        scripts_dir = osp.join(args.scripts_dir, 'train')
    else:
        scripts_dir = osp.join('./scripts', 'train')

    # create work_dir
    mkdir_or_exist(osp.abspath(scripts_dir))

    no_train_configs = filter_config(cfg, is_regeneration=args.is_regeneration, mode='train')

    base_master_port = 29647

    method_index = 0

    group_index = 0
    gpu_num = torch.cuda.device_count()
    no_train_configs = sorted(no_train_configs)

    fb_no_train_configs = []
    dl_no_train_configs = []
    for no_train_config in no_train_configs:
        if 'feature-based' in no_train_config:
            fb_no_train_configs.append(no_train_config)
        else:
            dl_no_train_configs.append(no_train_config)

    no_train_configs = dl_no_train_configs
    no_train_configs.extend(fb_no_train_configs)

    train_shs = [''] * args.group_num
    gpu_index = 0
    for config in no_train_configs:
        config_sh = ' ./configs/{}/{}'.format(config.split('_')[0], config + '.py')
        start_info = 'echo \"Start Train: {}\"\n\n'.format(config_sh)
        work_dir_sh = ' --work_dir {}'.format(cfg.work_dir)
        if 'feature-based' in config:
            python_sh = 'python tools/train_fb.py'
            end_sh = ' '
        else:
            if args.multi_gpu:
                python_sh = 'python -m torch.distributed.launch --nproc_per_node={} ' \
                            '--master_port={} tools/train.py'.format(gpu_num, base_master_port + method_index)
                end_sh = ' --seed 0 --launcher pytorch'
            else:
                python_sh = 'export CUDA_VISIBLE_DEVICES={} \npython tools/train.py'.format(gpu_index)
                end_sh = ' --seed 0'
                gpu_index += 1
                if gpu_index >= gpu_num:
                    gpu_index = 0
            method_index += 1

        if args.parallel:
            python_sh = python_sh.replace('python', 'nohup python')
            end_sh = f'{end_sh} > /dev/null 2>&1 &\n\n\n\n'
        train_shs[group_index] += start_info + python_sh + config_sh + work_dir_sh + end_sh + '\n\n'
        group_index += 1
        if group_index == args.group_num:
            group_index = 0

    for group_index, content in enumerate(train_shs):
        with open(os.path.join(scripts_dir, 'train_{:02d}.sh'.format(group_index)), 'w') as f:
            f.write(content)


if __name__ == '__main__':
    main()
