import argparse

from csrr.common.utils import Config, filter_config
from csrr.plots import build_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition Plot Results')
    parser.add_argument('config', help='plot config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Command Line Args:', args)

    # load cfg
    cfg = Config.fromfile(args.config)
    print(cfg.pretty_text)

    config_legend_map, config_method_map = filter_config(cfg, mode='summary')
    plot = build_plot(cfg.plot, cfg.log_dir, cfg.legend, cfg.scatter, config_legend_map, config_method_map)

    plot.plot()


if __name__ == '__main__':
    main()
