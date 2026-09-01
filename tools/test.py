import argparse
import hashlib
import json
import os
import os.path as osp
import pickle

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint


_DETECTION_MODEL_TYPES = frozenset(('SignalDetector', 'JDMFramework'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a model (classification or detection)')
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
              'to preserve the existing fast test path. Ignored for '
              'detection / joint configs.'))
    parser.add_argument(
        '--phase-rotation-tta-views',
        type=int,
        default=1,
        help=('average a fixed number of uniformly spaced carrier-phase '
              'views; MLDNN A/P semantics are read from the test pipeline'))
    args = parser.parse_args()
    return args


def _evaluator_is_detection(evaluator):
    if isinstance(evaluator, (list, tuple)):
        return any(_evaluator_is_detection(item) for item in evaluator)
    if not isinstance(evaluator, dict):
        return False
    name = str(evaluator.get('type', ''))
    return name in ('SignalDetectionMetric',) or 'Detection' in name


def _is_detection_cfg(cfg):
    model = cfg.get('model') or {}
    if isinstance(model, dict) and model.get('type') in _DETECTION_MODEL_TYPES:
        return True
    return _evaluator_is_detection(cfg.get('test_evaluator'))


def _set_default_snr_outputs(evaluator, work_dir):
    """Place SNR curve artifacts in the active work dir by default."""
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            _set_default_snr_outputs(item, work_dir)
        return
    if not isinstance(evaluator, dict) or not evaluator.get('snrwise', False):
        return
    evaluator.setdefault('snr_curve_out',
                         osp.join(work_dir, 'snr_curve.json'))
    evaluator.setdefault('snr_plot_out',
                         osp.join(work_dir, 'snr_curve.pdf'))


def _run_detection_test(args, cfg):
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    cfg.test_dataloader.setdefault('collate_fn', dict(type='default_collate'))
    _set_default_snr_outputs(cfg.test_evaluator, cfg.work_dir)

    runner = Runner.from_cfg(cfg)
    runner.test()


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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_order_from_test_pipeline(cfg):
    pipeline = cfg.test_dataloader.dataset.get('pipeline', [])
    transforms = [item for item in pipeline
                  if isinstance(item, dict)
                  and item.get('type') == 'MLDNNIQToAP']
    if len(transforms) != 1:
        raise ValueError(
            'phase-view evaluation requires exactly one MLDNNIQToAP in the '
            'resolved test pipeline')
    if 'phase_order' not in transforms[0]:
        raise ValueError(
            'MLDNNIQToAP.phase_order must be explicit for phase views')
    phase_order = transforms[0]['phase_order']
    if phase_order not in ('real_over_imag', 'imag_over_real'):
        raise ValueError(f'unsupported MLDNN phase_order: {phase_order!r}')
    return phase_order


def _rotate_mldnn_inputs(inputs, angle, phase_order):
    if not isinstance(inputs, dict) or 'iq' not in inputs or 'ap' not in inputs:
        raise TypeError('phase views require dict inputs containing iq and ap')
    iq = inputs['iq']
    if iq.ndim == 4 and iq.shape[1:3] == (1, 2):
        real, imag = iq[:, :, 0, :], iq[:, :, 1, :]
        channel_dim = 2
    elif iq.ndim == 3 and iq.shape[1] == 2:
        real, imag = iq[:, 0, :], iq[:, 1, :]
        channel_dim = 1
    else:
        raise ValueError(
            'phase views expect batched IQ shape (N,1,2,L) or (N,2,L)')

    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotated_real = cosine * real - sine * imag
    rotated_imag = sine * real + cosine * imag
    rotated_iq = torch.stack((rotated_real, rotated_imag), dim=channel_dim)
    amplitude = torch.sqrt(rotated_real.square() + rotated_imag.square())
    eps = torch.finfo(rotated_iq.dtype).eps
    if phase_order == 'real_over_imag':
        phase = torch.atan(rotated_real / (rotated_imag + eps))
    else:
        phase = torch.atan(rotated_imag / (rotated_real + eps))
    rotated_ap = torch.stack((amplitude, phase), dim=channel_dim)
    return dict(inputs, iq=rotated_iq, ap=rotated_ap)


def _test_step_with_phase_views(model, data, views, phase_order):
    if views == 1:
        return model.test_step(data)
    if views < 1:
        raise ValueError('--phase-rotation-tta-views must be positive')

    batches = []
    first_results = None
    for view in range(views):
        angle = 2.0 * np.pi * view / views
        view_data = dict(data)
        view_data['inputs'] = _rotate_mldnn_inputs(
            data['inputs'], angle, phase_order)
        results = model.test_step(view_data)
        if first_results is None:
            first_results = results
        batches.append(torch.stack([sample.pred_score for sample in results]))
    mean_scores = torch.stack(batches).mean(dim=0)
    for sample, score in zip(first_results, mean_scores):
        sample.set_pred_score(score)
        sample.set_pred_label(score.argmax().reshape(1))
    return first_results


def calculate_amc_metrics(pps, gts, snrs):
    predictions = np.asarray(pps).argmax(axis=1)
    labels = np.asarray(gts)
    snrs = np.asarray(snrs)
    if not (predictions.shape == labels.shape == snrs.shape):
        raise ValueError('prediction, label, and SNR counts must match')
    if not labels.size:
        raise ValueError('cannot evaluate an empty prediction set')
    per_snr = {}
    for snr in sorted(np.unique(snrs).tolist()):
        mask = snrs == snr
        per_snr[str(snr)] = float((predictions[mask] == labels[mask]).mean()
                                  * 100.0)
    pooled = float((predictions == labels).mean() * 100.0)
    maa = float(np.mean(list(per_snr.values())))
    return {
        'accuracy/top1': pooled,
        'accuracy/maa': maa,
        'accuracy/peak_snr': max(per_snr.values()),
        'accuracy/per_snr': per_snr,
    }


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if _is_detection_cfg(cfg):
        _run_detection_test(args, cfg)
        return

    if args.phase_rotation_tta_views < 1:
        raise ValueError('--phase-rotation-tta-views must be positive')
    phase_order = None
    if args.phase_rotation_tta_views > 1:
        phase_order = _phase_order_from_test_pipeline(cfg)

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

    # Match tools/train.py: default to ``default_collate`` so multi-input
    # models (e.g. MLDNN/FastMLDNN, whose pipeline packs an ``{iq, ap}`` dict)
    # get their per-key tensors stacked into batched tensors. ``pseudo_collate``
    # (mmengine's default) would leave a list-of-dicts that the data
    # preprocessor cannot stack. Single-input configs are unaffected.
    cfg.test_dataloader.setdefault('collate_fn', dict(type='default_collate'))
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
            results = _test_step_with_phase_views(
                model, data, args.phase_rotation_tta_views, phase_order)
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

    provenance = {
        'config': osp.relpath(osp.abspath(args.config)),
        'config_sha256': _sha256(args.config),
        'checkpoint_sha256': _sha256(args.checkpoint),
        'phase_rotation_tta_views': args.phase_rotation_tta_views,
        'phase_order': phase_order,
    }
    res = dict(pps=pps, gts=gts, snrs=snrs, classes=classes,
               provenance=provenance)

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

    metrics = calculate_amc_metrics(pps, gts, snrs)
    metrics_path = osp.join(work_dir, 'amc_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nResults saved to {save_path}')
    print(f'  metrics: {metrics_path}')
    print(f'  samples: {pps.shape[0]}, classes: {pps.shape[1]}')
    if 'feas' in res:
        print(f'  features: {res["feas"].shape}, centers: '
              f'{res["centers"].shape}')
    print(f'  pooled accuracy: {metrics["accuracy/top1"]:.4f}%')
    print(f'  average accuracy (MAA): {metrics["accuracy/maa"]:.4f}%')
    for snr, value in metrics['accuracy/per_snr'].items():
        print(f'    SNR {snr}: {value:.4f}%')


if __name__ == '__main__':
    main()
