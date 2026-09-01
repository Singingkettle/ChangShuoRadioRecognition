import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from mmengine.config import Config


def find_repo_root():
    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        if (parent / 'tools/train.py').is_file() and (parent / 'csrr').is_dir():
            return parent
    raise RuntimeError('could not locate the CSRR repository root')


REPO = find_repo_root()
PAPER_DIR = REPO / 'configs/mldnn'
sys.path.insert(0, str(PAPER_DIR))

from release_utils import (aggregate_predictions, calculate_metrics,
                           select_validation_epoch)  # noqa: E402


def load_test_module():
    spec = importlib.util.spec_from_file_location(
        'csrr_release_test_tool', REPO / 'tools/test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMLDNNReleaseProtocol(unittest.TestCase):

    def test_configs_enforce_two_stage_split(self):
        for dataset in ('201610a', '201801a'):
            calibration = Config.fromfile(
                PAPER_DIR / f'mldnn_iq-ap-deepsig-{dataset}.py')
            final = Config.fromfile(
                PAPER_DIR / 'experiments' /
                f'mldnn_iq-ap-deepsig-{dataset}_final.py')
            self.assertEqual(calibration.train_dataloader.dataset.ann_file,
                             'train.json')
            self.assertEqual(calibration.val_dataloader.dataset.ann_file,
                             'validation.json')
            self.assertIsNone(calibration.test_dataloader)
            self.assertEqual(final.train_dataloader.dataset.ann_file,
                             'train_and_validation.json')
            self.assertIsNone(final.val_dataloader)
            self.assertIsNone(final.val_evaluator)
            self.assertIsNone(final.val_cfg)
            self.assertEqual(final.test_dataloader.dataset.ann_file,
                             'test.json')

    def test_epoch_selection_uses_validation_top1_and_earliest_tie(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vis_data'
            path.mkdir()
            records = [
                {'epoch': 1, 'accuracy/top1': 60.0},
                {'epoch': 2, 'accuracy/top1': 61.0},
                {'epoch': 3, 'accuracy/top1': 61.0},
            ]
            with (path / 'scalars.json').open('w') as stream:
                for record in records:
                    stream.write(json.dumps(record) + '\n')
            self.assertEqual(select_validation_epoch(temporary), (2, 61.0))

    def test_maa_is_not_pooled_accuracy(self):
        scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        labels = np.array([0, 0, 1])
        snrs = np.array([0, 0, 10])
        metrics = calculate_metrics(scores, labels, snrs)
        self.assertAlmostEqual(metrics['accuracy/top1'], 200 / 3)
        self.assertEqual(metrics['accuracy/maa'], 50.0)
        tool_metrics = load_test_module().calculate_amc_metrics(
            scores, labels, snrs)
        self.assertEqual(tool_metrics, metrics)

    def test_probability_aggregation_checks_sample_order(self):
        with TemporaryDirectory() as temporary:
            paths = []
            for index in range(2):
                path = Path(temporary) / f'{index}.pkl'
                payload = {
                    'pps': np.eye(2), 'gts': np.array([0, 1]),
                    'snrs': np.array([-2, 0]), 'classes': ['A', 'B'],
                }
                import pickle
                with path.open('wb') as stream:
                    pickle.dump(payload, stream)
                paths.append(path)
            result = aggregate_predictions(paths)
            np.testing.assert_array_equal(result['pps'], np.eye(2))
            with paths[1].open('rb') as stream:
                import pickle
                changed = pickle.load(stream)
            changed['snrs'] = changed['snrs'][::-1]
            with paths[1].open('wb') as stream:
                pickle.dump(changed, stream)
            with self.assertRaisesRegex(ValueError, 'SNR order'):
                aggregate_predictions(paths)

    def test_phase_views_recompute_ap_with_declared_formula(self):
        tool = load_test_module()
        inputs = {
            'iq': torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
            'ap': torch.zeros(1, 1, 2, 2),
        }
        rotated = tool._rotate_mldnn_inputs(
            inputs, np.pi / 2, 'imag_over_real')
        torch.testing.assert_close(rotated['iq'][:, :, 0],
                                   torch.tensor([[[-3.0, -4.0]]]))
        expected_phase = torch.atan(
            rotated['iq'][:, :, 1] /
            (rotated['iq'][:, :, 0] + torch.finfo(torch.float32).eps))
        torch.testing.assert_close(rotated['ap'][:, :, 1], expected_phase)


if __name__ == '__main__':
    unittest.main()
