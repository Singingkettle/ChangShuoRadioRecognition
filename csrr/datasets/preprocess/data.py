import copy
import math

from ..builder import PREPROCESSES


@PREPROCESSES.register_module()
class FilterBySNR:
    def __init__(self, snr_set):
        self.snr_set = snr_set

    def __call__(self, data_infos):

        annotations = data_infos['annotations']
        save_annotations = []
        for annotation in annotations:
            if annotation['snr'] in self.snr_set:
                save_annotations.append(annotation)

        data_infos['annotations'] = save_annotations
        data_infos['snrs'] = sorted(self.snr_set)

        return data_infos


@PREPROCESSES.register_module()
class SampleByRatio:
    def __init__(self, sample_ratio, key_set=None):
        if key_set is None:
            key_set = ['modulation', 'snr']
        self.sample_ratio = sample_ratio
        self.key_set = key_set

    def __call__(self, data_infos):
        annotations = data_infos['annotations']

        statistical_infos = dict()
        for annotation in annotations:
            new_key = [None] * len(self.key_set)
            for i, key_val in enumerate(self.key_set):
                new_key[i] = annotation[key_val]
            new_key = tuple(new_key)
            if new_key in statistical_infos:
                statistical_infos[new_key].append(copy.deepcopy(annotation))
            else:
                statistical_infos[new_key] = [copy.deepcopy(annotation)]

        save_annotations = []
        for key_val in statistical_infos:
            save_num = math.ceil(len(statistical_infos[key_val]) * self.sample_ratio)
            save_annotations.extend(copy.deepcopy(statistical_infos[key_val][:save_num]))

        data_infos['annotations'] = save_annotations

        return data_infos
