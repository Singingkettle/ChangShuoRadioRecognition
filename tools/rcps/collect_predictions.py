import argparse
import os
import os.path as osp
import pickle

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint


SPLIT_TO_DATALOADER = {
    'train': 'train_dataloader',
    'validation': 'val_dataloader',
    'val': 'val_dataloader',
    'test': 'test_dataloader',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect probabilities, labels, and reliability metadata for RCPS analysis.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--split', choices=sorted(SPLIT_TO_DATALOADER), default='test')
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config file')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir or cfg.get(
        'work_dir', osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0]))
    dataloader_cfg = cfg[SPLIT_TO_DATALOADER[args.split]]

    init_default_scope(cfg.get('default_scope', 'csrr'))

    from csrr.registry import MODELS

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataloader = Runner.build_dataloader(dataloader_cfg)
    dataset = dataloader.dataset
    classes = list(dataset.CLASSES)

    all_pps, all_gts, all_snrs = [], [], []
    print(f'Collecting {args.split} predictions on {len(dataset)} samples ...')
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            results = model.test_step(data)
            for sample in results:
                all_pps.append(sample.pred_score.cpu().numpy())
                all_gts.append(sample.gt_label.item())
                idx = sample.get('sample_idx')
                if idx is not None:
                    info = dataset.get_data_info(idx)
                    all_snrs.append(info.get('snr', 0))
                else:
                    all_snrs.append(0)
            if (i + 1) % 50 == 0:
                print(f'  [{i + 1}/{len(dataloader)}]')

    res = dict(
        pps=np.stack(all_pps),
        gts=np.array(all_gts, dtype=np.int64),
        snrs=np.array(all_snrs),
        classes=classes,
        split=args.split,
    )

    if args.out:
        save_path = args.out
    else:
        save_path = osp.join(work_dir, 'predictions', f'{args.split}.pkl')
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)

    acc = np.mean(np.argmax(res['pps'], axis=1) == res['gts']) * 100
    print(f'\nResults saved to {save_path}')
    print(f'  samples: {res["pps"].shape[0]}, classes: {res["pps"].shape[1]}')
    print(f'  overall accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
