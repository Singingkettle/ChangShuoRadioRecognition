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
    parser.add_argument(
        '--save-features',
        action='store_true',
        help=('also save per-sample feature vectors (`feas`) and per-class '
              'centers (`centers`) into paper.pkl when the backbone or '
              'classifier exposes a `pre_logits` stage. Disabled by default '
              'to preserve the existing fast test path.'))
    args = parser.parse_args()
    return args


def _try_extract_features(model, inputs):
    """Best-effort feature extraction.

    Returns a numpy array ``[N, D]`` of features for the current batch, or
    ``None`` if the model cannot expose a feature stage.
    """
    try:
        feats = model.extract_feat(inputs, stage='pre_logits')
    except Exception:
        try:
            feats = model.extract_feat(inputs, stage='neck')
        except Exception:
            return None

    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if not isinstance(feats, torch.Tensor):
        return None
    if feats.dim() > 2:
        feats = feats.flatten(start_dim=1)
    return feats.detach().cpu().numpy()


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
    all_feas = [] if args.save_features else None

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

            if args.save_features:
                batch = model.data_preprocessor(data, training=False)
                inputs = batch['inputs']
                batch_feats = _try_extract_features(model, inputs)
                if batch_feats is None:
                    print('[test] backbone does not expose a feature stage; '
                          'disabling --save-features for the rest of the run.')
                    args.save_features = False
                    all_feas = None
                else:
                    all_feas.append(batch_feats)

            if (i + 1) % 50 == 0:
                print(f'  [{i + 1}/{len(dataloader)}]')

    pps = np.stack(all_pps)
    gts = np.array(all_gts, dtype=np.int64)
    snrs = np.array(all_snrs)

    res = dict(pps=pps, gts=gts, snrs=snrs, classes=classes)

    if all_feas:
        feas = np.concatenate(all_feas, axis=0)
        if feas.shape[0] != gts.shape[0]:
            print(f'[test] feature/label count mismatch ({feas.shape[0]} vs '
                  f'{gts.shape[0]}); dropping features.')
        else:
            num_classes = pps.shape[1]
            centers = np.zeros((num_classes, feas.shape[1]),
                               dtype=feas.dtype)
            for c in range(num_classes):
                mask = gts == c
                if mask.any():
                    centers[c] = feas[mask].mean(axis=0)
            res['feas'] = feas
            res['centers'] = centers

    save_dir = osp.join(work_dir, 'res')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'paper.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)

    acc = np.mean(np.argmax(pps, axis=1) == gts) * 100
    print(f'\nResults saved to {save_path}')
    print(f'  samples: {pps.shape[0]}, classes: {pps.shape[1]}')
    if 'feas' in res:
        print(f'  features: {res["feas"].shape}, centers: '
              f'{res["centers"].shape}')
    print(f'  overall accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
