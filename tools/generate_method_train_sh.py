import argparse
import os.path

import torch

from csrr import glob
from csrr.common.utils import Config, filter_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition Generate Test.sh File')
    parser.add_argument('method', help='performance config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    #
    config_dir = os.path.join('./configs', args.method)
    configs = glob(config_dir, '.py')

    base_master_port = 2900
    method_index = 0
    content = ''
    for config in configs:
        if 'plot' in config or 'data' in config or 'schedule' in config or 'default_runtime' in config:
            continue
        python_sh = 'python -m torch.distributed.launch --nproc_per_node={} --master_port={} tools/train.py '.format(
            torch.cuda.device_count(), base_master_port + method_index)
        end_sh = ' --seed 0 --launcher pytorch'
        start_info = 'echo \"Start Test: {}\"\n'.format(config)
        content += start_info + python_sh + config + end_sh + '\n\n'
        method_index += 1

    with open(os.path.join(config_dir, f'train_{args.method}.sh'), 'w') as f:
            f.write(content)

if __name__ == '__main__':
    main()
