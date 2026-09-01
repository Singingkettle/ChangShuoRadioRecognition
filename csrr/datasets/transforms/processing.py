from typing import Union

import numpy as np
import numpy.linalg as LA

from csrr.registry import TRANSFORMS
from .base import BaseTransform

Number = Union[int, float]


@TRANSFORMS.register_module()
class SelfNormalize(BaseTransform):
    """SelfNormalize the tensor.

    Args:
        norms (Dict(str, Dict)): Configs to make selfnormalize about input numpy array

            Every value in the norms is a dict, which includes:
                ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional Order of the norm (see table under ``Notes``).
                inf means numpy's `inf` object. The default is None.
                axis : {None, int, 2-tuple of ints}, optional.
                    If `axis` is an integer, it specifies the axis of `x` along which to
                    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
                    axes that hold 2-D matrices, and the matrix norms of these matrices
                    are computed.  If `axis` is None then either a vector norm (when `x`
                    is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
                    is None.
                keep_dims : bool, optional
                If this is set to True, the axes which are normed over are left in the
                result as dimensions with size one.  With this option the result will
                broadcast correctly against the original `x`.
    """

    def __init__(self, norms) -> None:
        self.norms = norms

    def transform(self, results: dict) -> dict:
        """Function to normalize tensors.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, key f'{data}_norm_cfg' key is added in to
            result dict.
        """

        def norm(x, cfg):
            x = x / LA.norm(x, **cfg)
            return x

        for key in self.norms:
            results[key] = norm(results[key], self.norms[key])
            results[f'{key}_norm_cfg'] = self.norms[key]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(norms={self.norms})'
        return repr_str


@TRANSFORMS.register_module()
class IQToAP(BaseTransform):
    """Convert IQ frame to AP frame.

    """

    def transform(self, results: dict) -> dict:
        """Function to convert iq frame to ap frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: results, key 'ap' is added in to result dict.
        """

        iq = results['iq'][0, :] + 1j * results['iq'][1, :]
        amp = np.abs(iq)
        amp = amp / LA.norm(amp, 2)
        ang = np.arctan2(results['iq'][1, :], results['iq'][0, :]) / np.pi

        results['ap'] = np.vstack((amp, ang))

        return results


@TRANSFORMS.register_module()
class DAENormalize(BaseTransform):
    """Normalize the ap frame
    """

    def transform(self, results: dict) -> dict:
        """Function to normalize ap frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: results, key 'ap' is added in to result dict.
        """
        ap = results['ap']
        ap[0, :] = ap[0, :] / np.linalg.norm(ap[0, :])
        ap[1, :] = -1 + 2 / (ap[1, :].max() - ap[1, :].min()) * (ap[1, :] - ap[1, :].min())
        results['ap'] = ap
        return results


@TRANSFORMS.register_module()
class MLDNNIQToAP(BaseTransform):
    """Convert IQ frame to AP frame.

    """

    _PHASE_ORDERS = ('real_over_imag', 'imag_over_real')

    def __init__(self, phase_order: str = 'real_over_imag') -> None:
        if phase_order not in self._PHASE_ORDERS:
            raise ValueError(
                f'phase_order must be one of {self._PHASE_ORDERS}, '
                f'but got {phase_order!r}')
        self.phase_order = phase_order

    def transform(self, results: dict) -> dict:
        """Function to convert iq frame to ap frame.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: results, key 'ap' is added in to result dict.
        """

        iq = results['iq'][0, :] + 1j * results['iq'][1, :]
        amp = np.abs(iq)
        real = results['iq'][0, :]
        imag = results['iq'][1, :]
        eps = np.finfo(np.float64).eps
        if self.phase_order == 'real_over_imag':
            ang = np.arctan(real / (imag + eps))
        else:
            ang = np.arctan(imag / (real + eps))
        results['ap'] = np.vstack((amp, ang))

        return results


@TRANSFORMS.register_module()
class MLDNNSNRLabel(BaseTransform):
    """Generate SNR label for MLDNN.

    """

    def transform(self, results: dict) -> dict:
        """Function to generate SNR label for MLDNN.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: results, key 'gt_label' is replaced with a dict for amc task and snr classification task.
        """

        if results['snr'] >= 0:
            snr_label = np.array(0, np.int64)
        else:
            snr_label = np.array(1, np.int64)
        results['gt_label'] = dict(amc=results['gt_label'], snr=snr_label)

        return results


@TRANSFORMS.register_module()
class SNRLabel(BaseTransform):
    """Generate SNR label.

    """

    def transform(self, results: dict) -> dict:
        """Function to generate SNR label.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: results, key 'gt_label' is replaced with a dict for amc task and snr classification task.
        """

        results['gt_label'] = dict(amc=results['gt_label'], snr=results['snr_label'])

        return results


@TRANSFORMS.register_module()
class RadioAugment(BaseTransform):
    """Label-preserving I/Q augmentation for modulation classification.

    Operates on the ``(2, L)`` baseband crop (``iq[0]`` = real, ``iq[1]`` =
    imag) produced by :class:`CSRDSignalToBaseband`. All three operations are
    modulation-invariant, so the modulation label is unchanged — they only
    enlarge the training distribution over nuisance parameters the receiver
    should be invariant to:

    - **phase rotation**: multiply by ``exp(j*theta)``; a random constant
      carrier phase, always present in real receivers.
    - **time shift**: circular roll along the sample axis; an unknown symbol
      timing offset.
    - **frequency offset**: multiply by ``exp(j*2*pi*f*n/L)`` for a small
      normalized ``f``; residual carrier-frequency offset after downconversion.

    These are standard, unpublished-but-benign training details (they do not
    change the architecture, the reported metrics, or the paper narrative).
    Applied only at training time; disable in val/test pipelines.

    Args:
        key (str): result key to augment. Defaults to ``'iq'``.
        phase (bool): enable random phase rotation. Defaults to True.
        time_shift (int): max absolute circular roll in samples (0 disables).
            Defaults to 0.
        freq_offset (float): max absolute normalized frequency offset (cycles
            over the whole window; 0 disables). Defaults to 0.0.
        prob (float): probability of applying the (whole) augmentation to a
            given sample. Defaults to 1.0.
    """

    def __init__(self, key: str = 'iq', phase: bool = True,
                 time_shift: int = 0, freq_offset: float = 0.0,
                 awgn_snr_db: tuple = None, awgn_prob: float = 0.5,
                 prob: float = 1.0) -> None:
        self.key = key
        self.phase = phase
        self.time_shift = int(time_shift)
        self.freq_offset = float(freq_offset)
        # (lo, hi) SNR range in dB for additive white Gaussian noise relative
        # to the crop's own power; None disables. Label-preserving: AWGN is
        # exactly the channel impairment the classifier must be invariant to.
        self.awgn_snr_db = tuple(awgn_snr_db) if awgn_snr_db else None
        self.awgn_prob = float(awgn_prob)
        self.prob = float(prob)

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        iq = results[self.key]
        c = iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)
        n = c.shape[-1]

        if self.phase:
            c = c * np.exp(1j * np.random.uniform(0.0, 2.0 * np.pi))
        if self.freq_offset > 0:
            f = np.random.uniform(-self.freq_offset, self.freq_offset)
            c = c * np.exp(1j * 2.0 * np.pi * f * np.arange(n) / n)
        if self.time_shift > 0:
            shift = int(np.random.randint(-self.time_shift,
                                          self.time_shift + 1))
            c = np.roll(c, shift)
        if self.awgn_snr_db is not None and np.random.rand() < self.awgn_prob:
            snr_db = np.random.uniform(*self.awgn_snr_db)
            sig_power = float(np.mean(np.abs(c) ** 2))
            if sig_power > 0:
                noise_power = sig_power / (10.0 ** (snr_db / 10.0))
                sigma = np.sqrt(noise_power / 2.0)
                c = c + (np.random.normal(0.0, sigma, n)
                         + 1j * np.random.normal(0.0, sigma, n))

        results[self.key] = np.stack([c.real, c.imag]).astype(np.float32)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(key={self.key!r}, '
                f'phase={self.phase}, time_shift={self.time_shift}, '
                f'freq_offset={self.freq_offset}, '
                f'awgn_snr_db={self.awgn_snr_db}, prob={self.prob})')

