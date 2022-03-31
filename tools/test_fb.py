import argparse

import numpy as np

from wtisp.common.utils import Config, mkdir_or_exist
from wtisp.datasets import build_dataset
from wtisp.models import build_fb


def parse_args():
    parser = argparse.ArgumentParser(
        description='WTISignalProcessing test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='model file')
    parser.add_argument(
        '--format-out',
        type=str,
        default=None,
        help='the dir to save output result file in json format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # format_out is determined in this priority: CLI > segment in file > filename
    if args.format_out is not None:
        # update configs according to CLI args if args.format_out is not None
        cfg.format_out = args.format_out
    elif cfg.get('format_out', None) is None:
        # use config filename as default format_out if cfg.format_out is None
        cfg.format_out = './format_out'

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    # build the model and load checkpoint
    cfg.model['model_path'] = args.model
    cfg.model['mode'] = 'test'
    model = build_fb(cfg.model)

    data = []
    for idx in range(len(dataset)):
        data.append(dataset[idx][cfg.x])

    data = np.concatenate(data)

    outputs = model(data)
    mkdir_or_exist(cfg.format_out)
    dataset.format_out(cfg.format_out, outputs)
    print(dataset.evaluate(outputs))


if __name__ == '__main__':
    main()
