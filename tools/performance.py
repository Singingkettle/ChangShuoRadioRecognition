import argparse

from csrr.common.utils import Config, filter_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition Plot Results')
    parser.add_argument('config', help='performance config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    config_legend_map, config_method_map = filter_config(cfg, mode='summary')
    pass



if __name__ == '__main__':
    main()
