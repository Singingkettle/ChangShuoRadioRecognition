from typing import Tuple

import numpy as np

from csrr.registry import DATA_FILTERS
from .base import BaseFilter


@DATA_FILTERS.register_module()
class FilterBySNR(BaseFilter):
    def __init__(self, save_range: Tuple[float, float]):
        self.save_range = save_range

    def filter(self, data_list, meta_info):
        new_data_list = []
        new_SNRs = []
        new_CLASSES = []

        for data in data_list:
            if self.save_range[0] <= data['snr'] <= self.save_range[1]:
                new_data_list.append(data)
                new_SNRs.append(data['snr'])
                new_CLASSES.append(data['modulation'])

        new_SNRs = sorted(list(set(new_SNRs)))
        new_CLASSES = sorted(list(set(new_CLASSES)))

        for idx, data in enumerate(new_data_list):
            gt_label = np.array(new_CLASSES.index(data['modulation']), dtype=np.int64)
            snr_label = np.array(new_SNRs.index(data['snr']), dtype=np.int64)
            new_data_list[idx]['gt_label'] = gt_label
            new_data_list[idx]['snr_label'] = snr_label

        meta_info['classes'] = new_CLASSES
        meta_info['snrs'] = new_SNRs
        meta_info['modulations'] = new_CLASSES

        return new_data_list, meta_info

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(save_range={self.save_range})'
        return repr_str