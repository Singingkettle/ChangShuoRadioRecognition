from csrr.registry import DATA_FILTERS
import math


@DATA_FILTERS.register_module()
class FilterByGroupRatio:
    def __init__(self, sample_ratio, group):
        self.sample_ratio = sample_ratio
        self.group = group

    def __call__(self, data_list):

        statistical_infos = dict()
        for data_info in data_list:
            new_key = [None] * len(self.group)
            for i, key_val in enumerate(self.group):
                new_key[i] = data_info[key_val]
            new_key = tuple(new_key)
            if new_key in statistical_infos:
                statistical_infos[new_key].append(data_info)
            else:
                statistical_infos[new_key] = [data_info]

        new_data_list = []
        for key_val in statistical_infos:
            save_num = math.ceil(len(statistical_infos[key_val]) * self.sample_ratio)
            new_data_list.extend(statistical_infos[key_val][:save_num])

        return new_data_list
