import argparse

from csrr.common.utils import Config
from csrr.performance.builder import build_performance


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition Performance Analysis')
    parser.add_argument('config', help='performance config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    cfg.performance['info'] = cfg.info
    performance = build_performance(cfg.performance)
    performance.draw()


if __name__ == '__main__':
    main()
