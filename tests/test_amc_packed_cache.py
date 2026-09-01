import json
import os
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from csrr.datasets.amc import AMCDataset


class TestAMCPackedCache(unittest.TestCase):

    @staticmethod
    def fixture(root, packed=True):
        root = Path(root)
        (root / 'iq').mkdir()
        samples = {
            'a.npy': np.arange(8, dtype=np.float32).reshape(2, 4),
            'b.npy': np.arange(8, 16, dtype=np.float32).reshape(2, 4),
        }
        for name, value in samples.items():
            np.save(root / 'iq' / name, value)
        annotation = {
            'metainfo': {'modulations': ['A'], 'snrs': [0]},
            'data_list': [
                {'file_name': 'b.npy', 'modulation': 'A', 'snr': 0},
                {'file_name': 'a.npy', 'modulation': 'A', 'snr': 0},
            ],
        }
        (root / 'train.json').write_text(json.dumps(annotation))
        if packed:
            (root / 'cache').mkdir()
            cache = {
                'iq': np.concatenate([samples['b.npy'], samples['a.npy']]),
                'index': {'b.npy': 0, 'a.npy': 1},
            }
            with (root / 'cache/train_iq.pkl').open('wb') as stream:
                pickle.dump(cache, stream, protocol=4)
        return samples

    def test_packed_cache_matches_source_samples(self):
        with TemporaryDirectory() as temporary:
            samples = self.fixture(temporary)
            dataset = AMCDataset(
                ann_file='train.json', data_root=temporary, cache=True,
                cache_file='auto', serialize_data=False)
            np.testing.assert_array_equal(
                dataset.get_data_info(0)['iq'], samples['b.npy'])
            np.testing.assert_array_equal(
                dataset.get_data_info(1)['iq'], samples['a.npy'])

    def test_local_auto_cache_can_fall_back_to_npy(self):
        with TemporaryDirectory() as temporary:
            samples = self.fixture(temporary, packed=False)
            dataset = AMCDataset(
                ann_file='train.json', data_root=temporary, cache=True,
                cache_file='auto', serialize_data=False)
            np.testing.assert_array_equal(
                dataset.get_data_info(0)['iq'], samples['b.npy'])

    def test_explicit_deployed_cache_is_required(self):
        with TemporaryDirectory() as temporary:
            self.fixture(temporary, packed=False)
            missing = str(Path(temporary) / 'missing-cache')
            with patch.dict(os.environ, {'CSRR_AMC_CACHE_DIR': missing}):
                with self.assertRaises(FileNotFoundError):
                    AMCDataset(
                        ann_file='train.json', data_root=temporary, cache=True,
                        cache_file='auto', serialize_data=False)


if __name__ == '__main__':
    unittest.main()
