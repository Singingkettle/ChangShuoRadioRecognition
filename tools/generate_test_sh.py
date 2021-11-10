import argparse

import torch

from wtisp.common.utils import Config, filter_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessing Generate Test.sh File')
    parser.add_argument('config', help='plot config file path')
    parser.add_argument('--is_regeneration', default=True, type=bool, help='is retest')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    test_sh_name = cfg.plot['config'] + '_test.sh'

    train_configs = filter_config(cfg, is_regeneration=args.is_regeneration, mode='test')

    base_master_port = 29547
    with open(test_sh_name, 'w') as f:
        method_index = 0
        for config, epoch in train_configs.items():
            if epoch == 0:
                continue
            if epoch == -1:
                python_sh = 'python tools/test_fb.py'
            else:
                python_sh = 'python -m torch.distributed.launch --nproc_per_node={} \
                --master_port={} tools/test.py'.format(
                    torch.cuda.device_count(), base_master_port + method_index)
            config_sh = './configs/{}/{}'.format(
                config.split('_')[0], config + '.py')

            if epoch == -1:
                checkpoint_sh = cfg.log_dir + '/{}/model.fb'.format(config)
            else:
                checkpoint_sh = cfg.log_dir + '/{}/epoch_{}.pth'.format(config, epoch)
            format_out_sh = cfg.log_dir + '/{}/format_out'.format(config)

            if epoch == -1:
                test_sh = python_sh + ' ' + config_sh + ' ' + checkpoint_sh + ' ' + \
                          '--format_out ' + format_out_sh + '\n\n\n'
            else:
                test_sh = python_sh + ' ' + config_sh + ' ' + checkpoint_sh + ' ' + \
                          '--format-out ' + format_out_sh + ' --launcher pytorch\n\n\n'
            start_info = 'echo \"Start Test: {}\"\n'.format(config_sh)
            f.write(start_info)
            f.write(test_sh)
            method_index += 1


if __name__ == '__main__':
    main()
