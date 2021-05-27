import argparse

import torch

from wtisp.common.utils import Config, DictAction
from wtisp.common.utils.flops_counter import get_model_complexity_info
from wtisp.models import build_task


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape_iq_ap',
        type=int,
        nargs='+',
        default=[1, 2, 128],
        help='input sequence data')
    parser.add_argument(
        '--shape_co',
        type=int,
        nargs='+',
        default=[1, 128, 128],
        help='input constellation data')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_shape = [(args.shape_iq_ap[0], args.shape_iq_ap[1], args.shape_iq_ap[2]),
                   (args.shape_iq_ap[0], args.shape_iq_ap[1], args.shape_iq_ap[2]),
                   (args.shape_co[0], args.shape_co[1], args.shape_co[2])]

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_task(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))
    # warm up
    for _ in range(100):
        _, _, _ = get_model_complexity_info(model, input_shape)
    flops, params, inference_time = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\nInference Time: {inference_time}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
