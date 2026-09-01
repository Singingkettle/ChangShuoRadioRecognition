"""Deterministic checkpoint and prediction helpers for HCGDNN."""

import copy
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch


FUSION_KEYS = ('head.cnn', 'head.gru1', 'head.gru2')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def select_validation_epoch(work_dir):
    candidates = []
    for scalar_path in sorted(Path(work_dir).rglob('scalars.json')):
        with scalar_path.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if 'accuracy/top1' in record and 'epoch' in record:
                    candidates.append((float(record['accuracy/top1']),
                                       int(record['epoch'])))
    if not candidates:
        raise RuntimeError(f'no validation accuracy found below {work_dir}')
    best_accuracy = max(item[0] for item in candidates)
    best_epoch = min(epoch for accuracy, epoch in candidates
                     if accuracy == best_accuracy)
    return best_epoch, best_accuracy


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError(f'checkpoint has no state_dict: {path}')
    return checkpoint


def dataset_signature(checkpoint):
    dataset_meta = checkpoint.get('meta', {}).get('dataset_meta')
    if not isinstance(dataset_meta, dict):
        raise ValueError('checkpoint has no dataset_meta')
    classes = tuple(dataset_meta.get('classes', ()))
    modulations = tuple(dataset_meta.get('modulations', ()))
    snrs = tuple(dataset_meta.get('snrs', ()))
    if not classes or classes != modulations or not snrs:
        raise ValueError('checkpoint class/SNR order is incomplete')
    return classes, snrs


def save_checkpoint_atomic(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    if temporary.exists():
        temporary.unlink()
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def average_checkpoints(paths, output_path):
    paths = [Path(path) for path in paths]
    if len(paths) != 3 or len(set(map(str, paths))) != 3:
        raise ValueError('HCGDNN requires exactly three distinct checkpoints')
    checkpoints = [load_checkpoint(path) for path in paths]
    signature = dataset_signature(checkpoints[0])
    if any(dataset_signature(item) != signature for item in checkpoints[1:]):
        raise ValueError('checkpoint class/SNR order differs')
    states = [item['state_dict'] for item in checkpoints]
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError('checkpoint state-dict key order differs')

    averaged = {}
    for key in keys:
        values = [state[key].detach().cpu() for state in states]
        reference = values[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype
               for value in values[1:]):
            raise ValueError(f'checkpoint tensor metadata differs: {key}')
        if reference.is_floating_point():
            value = torch.stack([item.to(torch.float64)
                                 for item in values]).mean(0)
            averaged[key] = value.to(reference.dtype)
        elif reference.is_complex():
            value = torch.stack([item.to(torch.complex128)
                                 for item in values]).mean(0)
            averaged[key] = value.to(reference.dtype)
        else:
            if any(not torch.equal(reference, item) for item in values[1:]):
                raise ValueError(f'non-floating checkpoint tensor differs: {key}')
            averaged[key] = reference.clone()

    output = copy.deepcopy(checkpoints[-1])
    output['state_dict'] = averaged
    output['meta'] = copy.deepcopy(output.get('meta', {}))
    output['meta']['hcgdnn_checkpoint_average'] = {
        'kind': 'equal_parameter_mean',
        'sources': [{'name': path.name, 'sha256': sha256(path)}
                    for path in paths],
    }
    save_checkpoint_atomic(output, output_path)
    return output['meta']['hcgdnn_checkpoint_average']


def transplant_fusion(source_path, target_path, output_path):
    source = load_checkpoint(source_path)
    target = load_checkpoint(target_path)
    if dataset_signature(source) != dataset_signature(target):
        raise ValueError('calibration/final class or SNR order differs')
    source_state = source['state_dict']
    target_state = target['state_dict']
    if tuple(source_state) != tuple(target_state):
        raise ValueError('calibration/final state-dict key order differs')
    weights = {}
    for key in FUSION_KEYS:
        if key not in source_state or key not in target_state:
            raise KeyError(f'missing fusion buffer: {key}')
        value = source_state[key].detach().cpu()
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f'invalid fusion buffer: {key}')
        weights[key] = float(value)
        target_state[key] = value.to(target_state[key].dtype).clone()
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-5):
        raise ValueError(f'fusion weights do not sum to one: {weights}')
    target['meta'] = copy.deepcopy(target.get('meta', {}))
    target['meta']['hcgdnn_fusion_transplant'] = {
        'source_sha256': sha256(source_path),
        'target_sha256': sha256(target_path),
        'weights': weights,
    }
    save_checkpoint_atomic(target, output_path)
    restored = load_checkpoint(output_path)
    for key, expected in target_state.items():
        if not torch.equal(restored['state_dict'][key], expected):
            raise RuntimeError(f'checkpoint verification failed: {key}')
    return target['meta']['hcgdnn_fusion_transplant']


def load_prediction(path):
    with Path(path).open('rb') as stream:
        result = pickle.load(stream)
    required = {'pps', 'gts', 'snrs', 'classes'}
    if required - set(result):
        raise ValueError(f'prediction file is incomplete: {path}')
    result['pps'] = np.asarray(result['pps'])
    result['gts'] = np.asarray(result['gts'])
    result['snrs'] = np.asarray(result['snrs'])
    if not (len(result['pps']) == len(result['gts']) == len(result['snrs'])):
        raise ValueError(f'prediction arrays have different lengths: {path}')
    return result


def aggregate_predictions(paths):
    predictions = [load_prediction(path) for path in paths]
    reference = predictions[0]
    for current in predictions[1:]:
        if not np.array_equal(current['gts'], reference['gts']):
            raise ValueError('prediction files have different label order')
        if not np.array_equal(current['snrs'], reference['snrs']):
            raise ValueError('prediction files have different SNR order')
        if list(current['classes']) != list(reference['classes']):
            raise ValueError('prediction files have different class order')
        if current['pps'].shape != reference['pps'].shape:
            raise ValueError('prediction score shapes differ')
    return {
        'pps': np.stack([item['pps'] for item in predictions]).mean(0),
        'gts': reference['gts'],
        'snrs': reference['snrs'],
        'classes': list(reference['classes']),
        'members': [str(Path(path)) for path in paths],
        'aggregation': 'equal_probability_mean',
    }


def calculate_metrics(scores, labels, snrs):
    predictions = np.asarray(scores).argmax(1)
    labels = np.asarray(labels)
    snrs = np.asarray(snrs)
    if not (predictions.shape == labels.shape == snrs.shape):
        raise ValueError('prediction, label, and SNR counts must match')
    per_snr = {}
    for snr in sorted(np.unique(snrs).tolist()):
        mask = snrs == snr
        per_snr[str(snr)] = float(
            (predictions[mask] == labels[mask]).mean() * 100.0)
    return {
        'accuracy/top1': float((predictions == labels).mean() * 100.0),
        'accuracy/maa': float(np.mean(list(per_snr.values()))),
        'accuracy/peak_snr': max(per_snr.values()),
        'accuracy/per_snr': per_snr,
    }


def dump_prediction(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        pickle.dump(result, stream, protocol=4)
