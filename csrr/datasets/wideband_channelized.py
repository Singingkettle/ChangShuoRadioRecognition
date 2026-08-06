# Copyright (c) Shuo Chang and contributors. Licensed under the Apache License, Version 2.0.
"""Wideband channelized-IQ recognition dataset.

Each sample is one detected signal, channelized back to complex baseband from a
wideband capture (mix to the box center frequency, low-pass to the box bandwidth,
decimate, crop/pad to a fixed length ``L``, energy-normalize). The cache stores
``(X, y)`` per split: ``X`` is ``[N, 2, L]`` real/imag IQ and ``y`` the 57-class
fine label. This is the recognition half of the "detection is easy, recognition
is hard" study; the detector produces the boxes, this dataset re-classifies the
IQ inside them.

The cache ``.npz`` files are produced by the ``build`` step of the return-to-IQ
pipeline (``np.savez(out, X=..., y=..., fs=...)``). Point ``data_root`` at the
directory that holds them and ``ann_file`` at the split file, e.g.
``train_L1024.npz``.
"""
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from csrr.registry import DATASETS
from .base_dataset import BaseClassificationDataset

# 57 fine classes, ordered exactly as the detector's COCO category ids
# (sorted by id). Keep this list in sync with the detection annotations so the
# integer labels in the cache line up with the class names.
WIDEBAND_57_CLASSES = (
    '1024qam', '128qam_cross', '16ask', '16fsk', '16gfsk', '16gmsk', '16msk',
    '16psk', '16qam', '2fsk', '2gfsk', '2gmsk', '2msk', '256qam', '32ask',
    '32psk', '32qam', '32qam_cross', '4ask', '4fsk', '4gfsk', '4gmsk', '4msk',
    '512qam_cross', '64ask', '64psk', '64qam', '8ask', '8fsk', '8gfsk', '8gmsk',
    '8msk', '8psk', 'am-dsb', 'am-dsb-sc', 'am-lsb', 'am-usb', 'bpsk',
    'chirpss', 'fm', 'lfm-data', 'lfm-radar', 'ofdm-1024', 'ofdm-1200',
    'ofdm-128', 'ofdm-180', 'ofdm-2048', 'ofdm-256', 'ofdm-300', 'ofdm-512',
    'ofdm-600', 'ofdm-64', 'ofdm-72', 'ofdm-900', 'ook', 'qpsk', 'tone',
)

# The multi-carrier (OFDM) classes, routed to the coarse head's "multi" branch.
WIDEBAND_MULTI_INDICES = tuple(
    i for i, n in enumerate(WIDEBAND_57_CLASSES) if 'ofdm' in n.lower())


@DATASETS.register_module()
class WidebandChannelizedDataset(BaseClassificationDataset):
    """Channelized-IQ crops with 57-class fine labels, read from an ``.npz`` cache.

    Args:
        ann_file (str): The ``.npz`` cache file name (joined with ``data_root``).
            Must hold arrays ``X`` (``[N, 2, L]``, float32) and ``y`` (``[N]``,
            int64).
        data_root (str): Directory holding the cache files. Defaults to ''.
        pipeline (Sequence): Processing pipeline. Typically a single
            ``PackInputs(input_key='iq')``. Defaults to an empty tuple.
        cache (bool): If True, hold the whole ``X`` array in memory and hand each
            sample a view; otherwise store an index and slice on access. Defaults
            to True (the arrays are small enough to keep resident).
    """

    METAINFO = {'classes': WIDEBAND_57_CLASSES}

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 cache: bool = True) -> None:
        self.cache = cache
        self._X = None
        # ndarray data_infos do not pickle efficiently into the shared buffer, so
        # keep serialize_data off by default for this dataset.
        super().__init__(ann_file, metainfo, data_root, filter_cfg, indices,
                         serialize_data, pipeline, test_mode, lazy_init,
                         max_refetch)

    def load_data_list(self) -> List[dict]:
        d = np.load(self.ann_file)
        X = d['X']
        y = d['y']
        if X.ndim != 3 or X.shape[1] != 2:
            raise ValueError(
                f'Expected X of shape [N, 2, L], got {X.shape} from '
                f'{self.ann_file}.')
        n_cls = len(self.METAINFO['classes'])
        if int(y.min()) < 0 or int(y.max()) >= n_cls:
            raise ValueError(
                f'Labels out of range [0, {n_cls}) in {self.ann_file}: '
                f'min={int(y.min())}, max={int(y.max())}.')

        if self.cache:
            self._X = np.ascontiguousarray(X.astype(np.float32))

        data_list = []
        for i in range(X.shape[0]):
            info = dict(gt_label=np.array(y[i], dtype=np.int64))
            if self.cache:
                info['iq'] = self._X[i]
            else:
                info['iq'] = X[i].astype(np.float32)
            data_list.append(info)
        return data_list
