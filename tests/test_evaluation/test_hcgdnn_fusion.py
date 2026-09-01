from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from csrr.evaluation.metrics.hcgdnn import (
    HCGDNNWeightsAccuracy,
    _by_optimization,
)
from csrr.models.backbones.hcgdnn import HCGDNN
from csrr.models.heads.hcgdnn_head import HCGDNNHead


class TestHCGDNNFusion(TestCase):

    @staticmethod
    def build_head():
        return HCGDNNHead(loss={
            'cnn': nn.Identity(),
            'gru1': nn.Identity(),
            'gru2': nn.Identity(),
        })

    def test_backbone_matches_paper_parameter_count(self):
        model = HCGDNN(num_classes=11)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 463557)

    def test_fusion_weights_survive_checkpoint_round_trip(self):
        head = self.build_head()
        expected = {'cnn': 0.2, 'gru1': 0.3, 'gru2': 0.5}
        head.set_weights(expected)
        restored = self.build_head()
        restored.load_state_dict(head.state_dict())
        for name, value in expected.items():
            self.assertAlmostEqual(getattr(restored, name).item(), value)

    def test_prediction_preserves_fused_probabilities(self):
        head = self.build_head().eval()
        head.set_weights({'cnn': 0.2, 'gru1': 0.3, 'gru2': 0.5})
        logits = {
            'cnn': torch.tensor([[2.0, -1.0], [0.2, 0.8]]),
            'gru1': torch.tensor([[0.5, 0.1], [1.1, -0.4]]),
            'gru2': torch.tensor([[-0.3, 1.4], [0.0, 0.7]]),
        }
        expected = sum(
            torch.softmax(logits[name], dim=1) * getattr(head, name)
            for name in logits)
        predictions = head.predict(logits)
        actual = torch.stack([sample.pred_score for sample in predictions])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(1), torch.ones(2))

    def test_optimization_uses_paper_initial_weights(self):
        captured = {}

        def fake_minimize(*args, **kwargs):
            captured['x0'] = np.asarray(args[1]).copy()
            return SimpleNamespace(x=captured['x0'])

        predictions = [
            np.array([[0.9, 0.1], [0.8, 0.2]]),
            np.array([[0.6, 0.4], [0.7, 0.3]]),
            np.array([[0.4, 0.6], [0.3, 0.7]]),
        ]
        with patch('csrr.evaluation.metrics.hcgdnn.minimize', fake_minimize):
            weights, _ = _by_optimization(
                predictions, np.array([0, 1]), 'trust-constr',
                ['cnn', 'gru1', 'gru2'])
        np.testing.assert_array_equal(captured['x0'], [0.0, 0.0, 1.0])
        self.assertEqual(weights, {'cnn': 0.0, 'gru1': 0.0, 'gru2': 1.0})

    def test_full_sample_objective_gradient_and_hessian(self):
        captured = {}

        def fake_minimize(objective, x0, **kwargs):
            captured['objective'] = objective
            captured['jacobian'] = kwargs['jac']
            captured['hessian'] = kwargs['hess']
            return SimpleNamespace(x=np.asarray(x0))

        predictions = [
            np.array([[0.335, 0.333, 0.332], [0.332, 0.335, 0.333]]),
            np.array([[0.334, 0.332, 0.334], [0.333, 0.334, 0.333]]),
            np.array([[0.333, 0.334, 0.333], [0.334, 0.333, 0.333]]),
        ]
        targets = np.array([0, 1])
        weights = np.asarray([0.2, 0.3, 0.5])
        with patch('csrr.evaluation.metrics.hcgdnn.minimize', fake_minimize):
            _by_optimization(
                predictions, targets, 'trust-constr',
                ['cnn', 'gru1', 'gru2'])

        eps = 1e-6
        gradient_fd = np.empty_like(weights)
        for index in range(len(weights)):
            offset = np.zeros_like(weights)
            offset[index] = eps
            gradient_fd[index] = (
                captured['objective'](weights + offset)
                - captured['objective'](weights - offset)) / (2 * eps)
        np.testing.assert_allclose(
            captured['jacobian'](weights), gradient_fd,
            rtol=1e-5, atol=1e-6)

        hessian_fd = np.empty((len(weights), len(weights)))
        for index in range(len(weights)):
            offset = np.zeros_like(weights)
            offset[index] = eps
            hessian_fd[:, index] = (
                captured['jacobian'](weights + offset)
                - captured['jacobian'](weights - offset)) / (2 * eps)
        np.testing.assert_allclose(
            captured['hessian'](weights), hessian_fd,
            rtol=1e-5, atol=1e-5)

    def test_metric_keys_match_hook_contract(self):
        metric = HCGDNNWeightsAccuracy()
        metric.results = [
            {'cnn_pred_score': torch.tensor([0.8, 0.2]),
             'gru1_pred_score': torch.tensor([0.7, 0.3]),
             'gru2_pred_score': torch.tensor([0.9, 0.1]),
             'gt_label': torch.tensor([0])},
            {'cnn_pred_score': torch.tensor([0.3, 0.7]),
             'gru1_pred_score': torch.tensor([0.4, 0.6]),
             'gru2_pred_score': torch.tensor([0.2, 0.8]),
             'gt_label': torch.tensor([1])},
        ]
        optimized = ({'cnn': 0.2, 'gru1': 0.3, 'gru2': 0.5},
                     np.array([[0.8, 0.2], [0.2, 0.8]]))
        with patch('csrr.evaluation.metrics.hcgdnn._by_optimization',
                   return_value=optimized):
            metrics = metric.evaluate(2)
        self.assertEqual(metrics['weights/cnn'], 0.2)
        self.assertEqual(metrics['weights/gru1'], 0.3)
        self.assertEqual(metrics['weights/gru2'], 0.5)
        self.assertEqual(metrics['accuracy/top1'], 100.0)
