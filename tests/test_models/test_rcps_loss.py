import pytest
import torch
import torch.nn.functional as F

from csrr.models.losses import (LabelSmoothingCrossEntropyLoss,
                                RCPSCrossEntropyLoss, build_rcps_targets)
from csrr.structures import DataSample


def _sample(label, snr):
    sample = DataSample().set_gt_label(label)
    sample.set_field(snr, 'snr', field_type='metainfo')
    return sample


def test_rcps_targets_endpoint_behavior():
    labels = torch.tensor([0, 1])
    snrs = torch.tensor([-20.0, 18.0])
    targets = build_rcps_targets(
        labels,
        snrs,
        num_classes=3,
        reliability_map=dict(type='linear', min=-20, max=18),
        epsilon=dict(type='power', max=1.0, gamma=1.0),
        base=dict(type='uniform'))

    assert torch.allclose(targets[0], torch.full((3,), 1.0 / 3.0))
    assert torch.allclose(targets[1], F.one_hot(torch.tensor(1), 3).float())
    assert torch.allclose(targets.sum(dim=1), torch.ones(2))


def test_rcps_retention_power_preserves_high_reliability_targets():
    labels = torch.tensor([0, 1, 2])
    snrs = torch.tensor([-20.0, 10.4, 18.0])
    targets = build_rcps_targets(
        labels,
        snrs,
        num_classes=3,
        reliability_map=dict(type='linear', min=-20, max=18),
        epsilon=dict(
            type='retention_power', max=1.0, gamma=1.0, retain_min=0.8),
        base=dict(type='uniform'))

    assert not torch.allclose(targets[0], F.one_hot(labels[0], 3).float())
    assert torch.allclose(targets[1], F.one_hot(labels[1], 3).float())
    assert torch.allclose(targets[2], F.one_hot(labels[2], 3).float())
    assert torch.allclose(targets.sum(dim=1), torch.ones(3))


def test_rcps_epsilon_zero_matches_hard_ce():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.5, -0.5]])
    labels = torch.tensor([0, 1])
    samples = [_sample(0, -20), _sample(1, 18)]
    loss = RCPSCrossEntropyLoss(
        reliability_map=dict(type='linear', min=-20, max=18),
        epsilon=dict(type='constant', value=0.0),
        base=dict(type='uniform'))

    assert torch.allclose(
        loss(logits, labels, data_samples=samples),
        F.cross_entropy(logits, labels))


def test_label_smoothing_equals_constant_uniform_rcps_target():
    labels = torch.tensor([0, 1])
    targets = build_rcps_targets(
        labels,
        torch.tensor([0.0, 1.0]),
        num_classes=3,
        reliability_map=dict(type='identity'),
        epsilon=dict(type='constant', value=0.2),
        base=dict(type='uniform'))
    expected = 0.8 * F.one_hot(labels, 3).float() + 0.2 / 3.0
    assert torch.allclose(targets, expected)

    logits = torch.randn(2, 3)
    loss = LabelSmoothingCrossEntropyLoss(smoothing=0.2)
    manual = -(expected * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    assert torch.allclose(loss(logits, labels), manual)


def test_rcps_requires_reliability_metadata():
    logits = torch.randn(1, 3)
    labels = torch.tensor([0])
    loss = RCPSCrossEntropyLoss()
    with pytest.raises(KeyError):
        loss(logits, labels, data_samples=[DataSample().set_gt_label(0)])
