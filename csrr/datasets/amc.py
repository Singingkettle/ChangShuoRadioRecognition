import os.path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import load

from csrr.registry import DATASETS
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
                 cache: bool = False) -> None:

        self.cache = cache

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

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        data_list = []
        for raw_data_info in raw_data_list:
            gt_label = np.array(self.CLASSES.index(raw_data_info['modulation']), dtype=np.int64)
            data_info = dict(gt_label=gt_label)
            data_path = os.path.join(self.data_root, 'iq', raw_data_info['file_name'])
            if self.cache:
                x = np.load(data_path)
                data_info['iq'] = x.astype(np.float32)
            else:
                data_info['iq_path'] = data_path
            data_list.append(data_info)

        return data_list
