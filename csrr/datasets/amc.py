import os.path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import load

from csrr.registry import DATASETS, DATA_FILTERS
from .base_dataset import BaseClassificationDataset


@DATASETS.register_module()
class AMCDataset(BaseClassificationDataset):
    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = '',
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 use_snr: Optional[dict] = None,
                 cache: bool = False,
                 cache_file: Optional[str] = None) -> None:

        self.cache = cache
        self.cache_file = cache_file
        self.use_snr = use_snr

        super().__init__(ann_file, metainfo, data_root, filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch)

    def load_data_list(self) -> List[dict]:
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        metainfo['classes'] = metainfo['modulations']
        raw_data_list = annotations['data_list']
        SNRs = metainfo['snrs']
        packed_iq = None
        packed_index = None
        packed_rows = None
        required_auto_cache = False
        if self.cache and self.cache_file is not None:
            cache_path = self.cache_file
            if cache_path == 'auto':
                split = os.path.splitext(os.path.basename(self.ann_file))[0]
                cache_root = os.environ.get(
                    'CSRR_AMC_CACHE_DIR')
                if cache_root:
                    required_auto_cache = True
                else:
                    cache_root = os.path.join(self.data_root, 'cache')
                cache_path = os.path.join(cache_root, f'{split}_iq.pkl')
            elif not os.path.isabs(cache_path):
                cache_path = os.path.join(self.data_root, cache_path)

            if os.path.isfile(cache_path):
                packed = load(cache_path)
                if not isinstance(packed, dict):
                    raise TypeError(
                        f'Packed IQ cache must be a dict: {cache_path}')
                packed_iq = packed.get('iq')
                packed_index = packed.get('index')
                if not isinstance(packed_iq, np.ndarray):
                    raise TypeError(
                        f'Packed IQ cache has no ndarray iq: {cache_path}')
                if not isinstance(packed_index, dict):
                    raise TypeError(
                        f'Packed IQ cache has no index dict: {cache_path}')
                if len(packed_index) != len(raw_data_list):
                    raise ValueError(
                        'Packed IQ cache/annotation length mismatch: '
                        f'{len(packed_index)} != {len(raw_data_list)}')
                if not packed_index or packed_iq.shape[0] % len(packed_index):
                    raise ValueError(
                        f'Packed IQ cache has invalid shape: {cache_path}')
                packed_rows = packed_iq.shape[0] // len(packed_index)
            elif self.cache_file != 'auto' or required_auto_cache:
                raise FileNotFoundError(cache_path)

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        data_list = []
        for raw_data_info in raw_data_list:
            gt_label = np.array(self.CLASSES.index(raw_data_info['modulation']), dtype=np.int64)
            snr = raw_data_info['snr']
            snr_label = np.array(SNRs.index(snr), dtype=np.int64)
            data_info = dict(gt_label=gt_label, snr_label=snr_label, snr=snr, modulation=raw_data_info['modulation'])
            data_path = os.path.join(self.data_root, 'iq', raw_data_info['file_name'])
            if self.cache:
                if packed_iq is None:
                    x = np.load(data_path)
                else:
                    file_name = raw_data_info['file_name']
                    if file_name not in packed_index:
                        raise KeyError(
                            f'Packed IQ cache is missing {file_name!r}')
                    index = packed_index[file_name]
                    if not isinstance(index, (int, np.integer)) or not (
                            0 <= index < len(packed_index)):
                        raise ValueError(
                            f'Invalid packed IQ index for {file_name!r}: '
                            f'{index!r}')
                    start = int(index) * packed_rows
                    x = packed_iq[start:start + packed_rows]
                data_info['iq'] = x.astype(np.float32, copy=False)
            else:
                data_info['iq_path'] = data_path
            data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        if self.filter_cfg is not None:
            if isinstance(self.filter_cfg, list):
                for cfg in self.filter_cfg:
                    f = DATA_FILTERS.build(cfg)
                    self.data_list, self._metainfo = f(self.data_list, self._metainfo)
            else:
                f = DATA_FILTERS.build(self.filter_cfg)
                self.data_list, self._metainfo = f(self.data_list, self._metainfo)
        return self.data_list

