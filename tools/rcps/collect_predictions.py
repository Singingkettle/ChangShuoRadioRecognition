import argparse
import os
import os.path as osp
import pickle
from copy import deepcopy

import numpy as np
import torch
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


SPLIT_TO_DATALOADER = {
    'train': 'train_dataloader',
    'validation': 'val_dataloader',
    'val': 'val_dataloader',
    'test': 'test_dataloader',
}


def set_default_dataloader_cfg(cfg, field):
    """Mirror tools/train.py dataloader defaults for standalone export."""
    if cfg.get(field, None) is None:
        return
    dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        dataloader_cfg.persistent_workers = False
    dataloader_cfg.update(deepcopy(cfg[field]))
    cfg[field] = dataloader_cfg


def needs_mldnn_probability_recovery(cfg):
    """Return whether MLDNN predictions need double-softmax recovery."""
    model_cfg = cfg.get('model', {})
    backbone_type = model_cfg.get('backbone', {}).get('type', '')
    head_type = model_cfg.get('head', {}).get('type', '')
    return backbone_type == 'MLDNN' and head_type == 'MLDNNHead'


def recover_mldnn_merge_probability(score):
    """Undo the MLDNNHead softmax applied to an already-probabilistic merge.

    The current MLDNN backbone returns ``merge`` as a convex combination of
    AP/IQ softmax probabilities at evaluation time. ``MLDNNHead.predict`` then
    applies another softmax before storing ``pred_score``. Accuracy is nearly
    unchanged by this monotone transform, but confidence, NLL, ECE, and Brier
    become systematically under-confident. If q = softmax(p) and sum(p)=1,
    then p_i = log(q_i) + c with c chosen so the recovered vector sums to one.
    """
    score = np.asarray(score, dtype=np.float64)
    log_score = np.log(np.clip(score, 1e-12, 1.0))
    recovered = log_score + (1.0 - log_score.sum()) / log_score.size
    recovered = np.clip(recovered, 0.0, None)
    return recovered / np.clip(recovered.sum(), 1e-12, None)


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
    set_default_dataloader_cfg(cfg, SPLIT_TO_DATALOADER[args.split])

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
    recover_mldnn_probs = needs_mldnn_probability_recovery(cfg)
    if recover_mldnn_probs:
        print('Recovering MLDNN merge probabilities from double-softmax pred_score.')
    print(f'Collecting {args.split} predictions on {len(dataset)} samples ...')
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            results = model.test_step(data)
            for sample in results:
                score = sample.pred_score.cpu().numpy()
                if recover_mldnn_probs:
                    score = recover_mldnn_merge_probability(score)
                all_pps.append(score)
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
