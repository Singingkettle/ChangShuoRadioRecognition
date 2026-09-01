import json
import pickle
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
PAPER_DIR = REPO / 'configs/hcgdnn'
sys.path.insert(0, str(PAPER_DIR))

from release_utils import (aggregate_predictions, average_checkpoints,
                           calculate_metrics, load_checkpoint,
                           select_validation_epoch,
                           transplant_fusion)  # noqa: E402


CLASSES = ('8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK',
           '4PAM', '16QAM', '64QAM', 'QPSK', 'WBFM')
SNRS = tuple(range(-20, 20, 2))


def checkpoint(path, value, fusion=(0.0, 0.0, 1.0), classes=CLASSES):
    payload = {
        'meta': {'dataset_meta': {
            'classes': list(classes), 'modulations': list(classes),
            'snrs': list(SNRS)}},
        'state_dict': {
            'weight': torch.tensor([value], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
            'head.cnn': torch.tensor(fusion[0]),
            'head.gru1': torch.tensor(fusion[1]),
            'head.gru2': torch.tensor(fusion[2]),
        },
    }
    torch.save(payload, path)


class TestHCGDNNReleaseProtocol(unittest.TestCase):

    def test_configs_enforce_two_stage_split_and_retention(self):
        calibration = Config.fromfile(
            PAPER_DIR / 'hcgdnn_iq-deepsig-201610a.py')
        final = Config.fromfile(
            PAPER_DIR / 'experiments/hcgdnn_iq-deepsig-201610a_final.py')
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
        self.assertEqual(final.test_dataloader.dataset.ann_file, 'test.json')
        checkpoint_hook = final.default_hooks.checkpoint
        self.assertEqual(checkpoint_hook.interval, 10)
        self.assertEqual(checkpoint_hook.max_keep_ckpts, 3)
        self.assertTrue(checkpoint_hook.save_last)

    def test_epoch_selection_uses_earliest_validation_tie(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vis_data'
            path.mkdir()
            with (path / 'scalars.json').open('w') as stream:
                for record in ({'epoch': 1, 'accuracy/top1': 60.0},
                               {'epoch': 2, 'accuracy/top1': 61.0},
                               {'epoch': 3, 'accuracy/top1': 61.0}):
                    stream.write(json.dumps(record) + '\n')
            self.assertEqual(select_validation_epoch(temporary), (2, 61.0))

    def test_checkpoint_average_and_fusion_transplant(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for index, value in enumerate((1.0, 2.0, 3.0)):
                path = root / f'epoch_{index}.pth'
                checkpoint(path, value)
                sources.append(path)
            calibration = root / 'calibration.pth'
            checkpoint(calibration, 99.0, fusion=(0.2, 0.3, 0.5))
            averaged = root / 'averaged.pth'
            final = root / 'final.pth'
            average_checkpoints(sources, averaged)
            transplant_fusion(calibration, averaged, final)
            state = load_checkpoint(final)['state_dict']
            torch.testing.assert_close(state['weight'], torch.tensor([2.0]))
            self.assertAlmostEqual(float(state['head.cnn']), 0.2)
            self.assertAlmostEqual(float(state['head.gru1']), 0.3)
            self.assertAlmostEqual(float(state['head.gru2']), 0.5)

    def test_checkpoint_class_order_mismatch_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(3):
                path = root / f'{index}.pth'
                classes = CLASSES if index < 2 else CLASSES[::-1]
                checkpoint(path, float(index), classes=classes)
                paths.append(path)
            with self.assertRaisesRegex(ValueError, 'class/SNR order'):
                average_checkpoints(paths, root / 'out.pth')

    def test_prediction_aggregation_checks_order_and_maa(self):
        with TemporaryDirectory() as temporary:
            paths = []
            for index in range(2):
                path = Path(temporary) / f'{index}.pkl'
                with path.open('wb') as stream:
                    pickle.dump({
                        'pps': np.array([[0.9, 0.1], [0.8, 0.2],
                                         [0.7, 0.3]]),
                        'gts': np.array([0, 0, 1]),
                        'snrs': np.array([0, 0, 10]),
                        'classes': ['A', 'B'],
                    }, stream)
                paths.append(path)
            result = aggregate_predictions(paths)
            metrics = calculate_metrics(
                result['pps'], result['gts'], result['snrs'])
            self.assertAlmostEqual(metrics['accuracy/top1'], 200 / 3)
            self.assertEqual(metrics['accuracy/maa'], 50.0)
            with paths[1].open('rb') as stream:
                changed = pickle.load(stream)
            changed['gts'] = changed['gts'][::-1]
            with paths[1].open('wb') as stream:
                pickle.dump(changed, stream)
            with self.assertRaisesRegex(ValueError, 'label order'):
                aggregate_predictions(paths)


if __name__ == '__main__':
    unittest.main()
