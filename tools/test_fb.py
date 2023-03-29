import argparse
import os
import time

from csrr.common.fileio import dump as IODump
from csrr.common.utils import Config, setup_multi_processes
from csrr.datasets import build_dataset
from csrr.models import build_method


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChangShuoRadioRecognition test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        help='the directory to format the file containing evaluation figures, results in pickle format, '
             'and format data to upload into test server')
    parser.add_argument('--out', action='store_true', help='output result file in pickle format')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        action='store_true',
        help='evaluation on the dataset')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only, \
        ('Please specify at least one operation (format/eval/format the '
         'results / format the results) with the argument "--out", "--eval"'
         ', "--format-only"')

    cfg = Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    inputs = []
    for signal in dataset:
        inputs.append(signal['inputs'][cfg.x])

    # build the model and load checkpoint
    cfg.model['model_path'] = args.checkpoint
    cfg.model['mode'] = 'test'
    model = build_method(cfg.model)
    outputs = model(inputs)

    # allows not to create
    # mkdir_or_exist(os.path.abspath(args.work_dir))

    if args.out:
        print(f'\nwriting results to {args.work_dir}')
        IODump(outputs, os.path.join(args.work_dir, 'results.pkl'))
    if args.format_only:
        dataset.format_results(args.work_dir, outputs)
    if args.eval:
        metric = dataset.evaluate(outputs)
        print(metric)
        metric_dict = dict(config=args.config, metric=metric)
        if args.work_dir is not None:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = os.path.join(args.work_dir, f'eval_{timestamp}.json')
            IODump(metric_dict, json_file)


if __name__ == '__main__':
    main()
