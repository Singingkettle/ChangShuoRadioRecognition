"""Small, deterministic helpers used by the public MLDNN workflow."""

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


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
                if 'accuracy/top1' not in record or 'epoch' not in record:
                    continue
                candidates.append((float(record['accuracy/top1']),
                                   int(record['epoch'])))
    if not candidates:
        raise RuntimeError(f'no validation accuracy found below {work_dir}')
    best_accuracy = max(item[0] for item in candidates)
    best_epoch = min(epoch for accuracy, epoch in candidates
                     if accuracy == best_accuracy)
    return best_epoch, best_accuracy


def load_prediction(path):
    with Path(path).open('rb') as stream:
        result = pickle.load(stream)
    required = {'pps', 'gts', 'snrs', 'classes'}
    missing = required - set(result)
    if missing:
        raise ValueError(f'prediction file lacks {sorted(missing)}: {path}')
    result['pps'] = np.asarray(result['pps'])
    result['gts'] = np.asarray(result['gts'])
    result['snrs'] = np.asarray(result['snrs'])
    if result['pps'].ndim != 2:
        raise ValueError(f'prediction scores must be two-dimensional: {path}')
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
            raise ValueError('prediction files have different score shapes')
    mean_scores = np.stack([item['pps'] for item in predictions]).mean(axis=0)
    return {
        'pps': mean_scores,
        'gts': reference['gts'],
        'snrs': reference['snrs'],
        'classes': list(reference['classes']),
        'members': [str(Path(path)) for path in paths],
        'aggregation': 'equal_probability_mean',
    }


def calculate_metrics(scores, labels, snrs):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    snrs = np.asarray(snrs)
    predictions = scores.argmax(axis=1)
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
