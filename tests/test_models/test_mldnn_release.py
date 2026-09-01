import unittest

import numpy as np
import torch

from csrr.datasets.transforms.processing import MLDNNIQToAP
from csrr.models.backbones.mldnn import MLDNN


class TestMLDNNRelease(unittest.TestCase):

    def test_phase_definitions_are_explicit(self):
        iq = np.array([[1.0, -2.0, 0.5], [2.0, 4.0, -1.0]])
        for phase_order, expected in (
                ('real_over_imag', np.arctan(iq[0] / iq[1])),
                ('imag_over_real', np.arctan(iq[1] / iq[0]))):
            result = MLDNNIQToAP(phase_order=phase_order)(
                {'iq': iq.copy()})
            np.testing.assert_allclose(result['ap'][0], np.abs(
                iq[0] + 1j * iq[1]))
            np.testing.assert_allclose(result['ap'][1], expected)
        with self.assertRaises(ValueError):
            MLDNNIQToAP(phase_order='ambiguous')

    def test_log_mixture_matches_paper_probability(self):
        model = MLDNN(
            num_classes=11, dropout_rate=0.0, use_GRU=True,
            is_BIGRU=True, fusion_method='safn', gradient_truncation=True,
            merge_log_probability=True)
        model.train()
        inputs = {
            'iq': torch.randn(2, 1, 2, 128),
            'ap': torch.randn(2, 1, 2, 128),
        }
        merge, ap, iq, snr = model(inputs)
        expected = (ap.softmax(1) * snr.softmax(1)[:, :1]
                    + iq.softmax(1) * snr.softmax(1)[:, 1:])
        torch.testing.assert_close(merge.exp(), expected)
        torch.testing.assert_close(
            merge.exp().sum(1), torch.ones(2), atol=1e-6, rtol=1e-6)

    def test_128_and_1024_forward_shapes(self):
        for length, classes, avg_pool in ((128, 11, None),
                                          (1024, 24, (1, 8))):
            model = MLDNN(
                num_classes=classes, dropout_rate=0.0, avg_pool=avg_pool,
                use_GRU=True, is_BIGRU=True, fusion_method='safn',
                merge_log_probability=True)
            model.eval()
            with torch.no_grad():
                output = model({
                    'iq': torch.randn(1, 1, 2, length),
                    'ap': torch.randn(1, 1, 2, length),
                })
            self.assertEqual(output[0].shape, (1, classes))


if __name__ == '__main__':
    unittest.main()
