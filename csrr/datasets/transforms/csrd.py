# Copyright (c) Shuo Chang. All Rights Reserved.
"""Transforms for CSRD (CRML23) multi-signal frames used by the JDM method."""
import numpy as np
import torch
from scipy.io import loadmat

from csrr.registry import TRANSFORMS
from csrr.structures import DataSample
from .base import BaseTransform
from .formatting import to_tensor


@TRANSFORMS.register_module()
class LoadCSRDFrame(BaseTransform):
    """Load a CSRD ``.mat`` entry.

    The file stores ``signal_data`` of shape ``(num_signals, 2, L)`` — the
    per-signal passband I/Q components — and, for noisy configurations
    (awgn-*/real/real_awgn-*), ``wideband_data`` of shape ``(1, 2, L)``:
    the received frame with the frame's single AWGN realization applied
    once at the wideband level.

    ``wideband_data`` is preferred as the received frame whenever present.
    Summing ``signal_data`` is only correct for noise-free entries; with
    older exports whose per-signal components each embedded the (same)
    wideband noise vector, summing stacked that noise ``num_signals``
    times, silently degrading the effective SNR far below the label
    (see docs/csrd_jointdet/dataset_generation.md).

    **Added keys**: ``iq`` (float32, shape (2, L)); optionally
    ``signal_components`` (float32, shape (num_signals, 2, L)).

    Args:
        keep_components (bool): also expose the raw per-signal components
            (needed by :class:`CSRDSignalToBaseband` with
            ``source='component'``). Defaults to False.
    """

    def __init__(self, keep_components: bool = False):
        self.keep_components = keep_components

    def transform(self, results: dict) -> dict:
        mat = loadmat(results['iq_path'])
        components = np.asarray(mat['signal_data'], dtype=np.float32)
        if components.ndim == 2:  # single-signal entry stored squeezed
            components = components[None, ...]
        if 'wideband_data' in mat:
            results['iq'] = np.asarray(
                mat['wideband_data'], dtype=np.float32)[0]
        else:
            results['iq'] = components.sum(axis=0)
        if self.keep_components:
            results['signal_components'] = components
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'keep_components={self.keep_components})'


@TRANSFORMS.register_module()
class IQToSpectrum(BaseTransform):
    """Convert a time-domain I/Q frame to its frequency representation.

    Computes the fftshift-ed FFT of ``iq`` and stacks amplitude and phase,
    matching the input of the JDM detection module (and the historical
    ``LoadFFTofCSRR`` transform).

    **Modified keys**: writes ``spectrum`` (float32, shape (2, L)).

    Args:
        to_norm (bool): per-channel standardization of the output.
            Defaults to False (historical configs did not normalize).
    """

    def __init__(self, to_norm: bool = False):
        self.to_norm = to_norm

    def transform(self, results: dict) -> dict:
        iq = results['iq']
        frame = iq[0] + 1j * iq[1]
        spectrum = np.fft.fftshift(np.fft.fft(frame))
        spectrum = np.stack([np.abs(spectrum), np.angle(spectrum)])
        if self.to_norm:
            spectrum = (spectrum - spectrum.mean(axis=1, keepdims=True)) \
                / (spectrum.std(axis=1, keepdims=True) + 1e-12)
        results['spectrum'] = spectrum.astype(np.float32)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_norm={self.to_norm})'


@TRANSFORMS.register_module()
class CSRDSignalToBaseband(BaseTransform):
    """Extract the single-signal baseband crop of one annotated signal.

    Mirrors the proposal filtering of the JDM classification module
    (paper Sec. V-C): keep only the FFT bins inside the signal band
    (ideal low-pass), roll the band center to DC (carrier removal) and go
    back to the time domain.

    **Required keys**: ``iq`` (and ``signal_components`` when
    ``source='component'``), ``signal_index``, ``center_frequency``,
    ``bandwidth``, ``sample_rate``.
    **Modified keys**: ``iq`` — replaced by the (2, L) baseband crop.

    Args:
        source (str): ``'frame'`` filters the received (summed) frame, which
            matches inference conditions including adjacent-signal leakage;
            ``'component'`` filters the signal's own passband component only.
            Defaults to 'frame'.
    """

    def __init__(self, source: str = 'frame'):
        assert source in ('frame', 'component')
        self.source = source

    def transform(self, results: dict) -> dict:
        if self.source == 'frame':
            iq = results['iq']
        else:
            iq = results['signal_components'][results['signal_index']]
        frame = iq[0] + 1j * iq[1]
        num_bins = frame.shape[-1]
        spectrum = np.fft.fftshift(np.fft.fft(frame))

        sample_rate = results['sample_rate']
        cf, bw = results['center_frequency'], results['bandwidth']
        left = int(round(((cf - bw / 2) / sample_rate + 0.5) * num_bins))
        right = int(round(((cf + bw / 2) / sample_rate + 0.5) * num_bins))
        left = min(max(left, 0), num_bins - 1)
        right = min(max(right, left + 1), num_bins)

        masked = np.zeros_like(spectrum)
        masked[left:right] = spectrum[left:right]
        masked = np.roll(masked, num_bins // 2 - (left + right) // 2)
        baseband = np.fft.ifft(np.fft.ifftshift(masked))
        results['iq'] = np.stack(
            [baseband.real, baseband.imag]).astype(np.float32)
        results.pop('signal_components', None)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(source={self.source!r})'


@TRANSFORMS.register_module()
class PackDetectionInputs(BaseTransform):
    """Pack a detection sample into model inputs + :class:`DataSample`.

    The data sample carries ``gt_boxes`` — 1-D frequency intervals
    ``(left, right)`` in FFT-bin units, shape (num_signals, 2) — and
    ``gt_box_labels`` (modulation indices, shape (num_signals,)).

    Args:
        input_key (str): key of the input array ('spectrum' for the
            detector, 'iq' for the end-to-end JDM framework).
        meta_keys (Sequence[str]): keys stored as metainfo.
    """

    DEFAULT_META_KEYS = ('sample_idx', 'file_name', 'version', 'snr',
                         'channel', 'frame_length', 'sample_rate')

    def __init__(self,
                 input_key: str = 'spectrum',
                 meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict(
            inputs=to_tensor(results[self.input_key]).contiguous())

        data_sample = DataSample()
        data_sample.set_field(
            to_tensor(results['gt_boxes']).to(torch.float32).reshape(-1, 2),
            'gt_boxes')
        data_sample.set_field(
            to_tensor(results['gt_box_labels']).to(torch.long).reshape(-1),
            'gt_box_labels')
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key,
                                      field_type='metainfo')
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        return f'{self.__class__.__name__}(input_key={self.input_key!r}, ' \
               f'meta_keys={self.meta_keys})'
