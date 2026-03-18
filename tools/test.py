import argparse
import os
import os.path as osp
import pickle

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a model and save predictions for plotting')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        work_dir = args.work_dir
    elif cfg.get('work_dir', None) is not None:
        work_dir = cfg.work_dir
    else:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

    init_default_scope(cfg.get('default_scope', 'csrr'))

    from csrr.registry import MODELS

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    dataset = dataloader.dataset
    classes = list(dataset.CLASSES)

    all_pps = []
    all_gts = []
    all_snrs = []

    print(f'Collecting predictions on {len(dataset)} samples ...')

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

    pps = np.stack(all_pps)
    gts = np.array(all_gts, dtype=np.int64)
    snrs = np.array(all_snrs)

    res = dict(pps=pps, gts=gts, snrs=snrs, classes=classes)

    save_dir = osp.join(work_dir, 'res')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'paper.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)

    acc = np.mean(np.argmax(pps, axis=1) == gts) * 100
    print(f'\nResults saved to {save_path}')
    print(f'  samples: {pps.shape[0]}, classes: {pps.shape[1]}')
    print(f'  overall accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
