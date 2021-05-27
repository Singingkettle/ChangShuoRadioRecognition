import argparse

from wtisp.common.utils import Config
from wtisp.plot import build_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessing Plot Results')
    parser.add_argument('config', help='plot config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    plot = build_plot(cfg.plot)

    plot.plot()


if __name__ == '__main__':
    main()
