import json

import numpy as np
import pytest

from csrr.evaluation.metrics.detection import SignalDetectionMetric


def test_signal_detection_metric_groups_by_per_box_snr(tmp_path):
    curve_path = tmp_path / 'snr_curve.json'
    metric = SignalDetectionMetric(
        snrwise=True, snr_curve_out=str(curve_path))

    metric.process(None, [dict(
        pred_boxes=np.array([[0.0, 10.0]], dtype=np.float32),
        pred_box_scores=np.array([0.9], dtype=np.float32),
        gt_boxes=np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32),
        gt_box_labels=np.array([0, 0], dtype=np.int64),
        snr=np.array([0, 10], dtype=np.int64),
    )])
    metrics = metric.compute_metrics(metric.results)

    assert metrics['mAP_snr_0'] == 1.0
    assert metrics['AR_snr_0'] == 1.0
    assert metrics['mAP_snr_10'] == 0.0
    assert metrics['AR_snr_10'] == 0.0

    curve = json.loads(curve_path.read_text())['points']
    assert [point['snr'] for point in curve] == [0, 10]
    assert [point['num_gt'] for point in curve] == [1, 1]


def test_signal_detection_metric_rejects_frame_level_snr():
    metric = SignalDetectionMetric(snrwise=True)

    with pytest.raises(ValueError, match='Frame-level SNR cannot be used'):
        metric.process(None, [dict(
            pred_boxes=np.zeros((0, 2), dtype=np.float32),
            pred_box_scores=np.zeros((0, ), dtype=np.float32),
            gt_boxes=np.array([[0.0, 10.0], [20.0, 30.0]],
                              dtype=np.float32),
            gt_box_labels=np.array([0, 0], dtype=np.int64),
            snr=np.array([0], dtype=np.int64),
        )])


def test_signal_detection_metric_accepts_string_snr_labels():
    metric = SignalDetectionMetric(snrwise=True)

    metric.process(None, [dict(
        pred_boxes=np.array([[0.0, 10.0]], dtype=np.float32),
        pred_box_scores=np.array([0.9], dtype=np.float32),
        gt_boxes=np.array([[0.0, 10.0]], dtype=np.float32),
        gt_box_labels=np.array([0], dtype=np.int64),
        snr=np.array(['infdB']),
    )])
    metrics = metric.compute_metrics(metric.results)

    assert metrics['mAP_snr_infdB'] == 1.0
