import copy
import math

from ..builder import PREPROCESSES


@PREPROCESSES.register_module()
class FilterBySNR:
    def __init__(self, snr_set):
        self.snr_set = snr_set

    def __call__(self, data_infos):
        item_filename = []
        item_mod_label = []
        item_snr_value = data_infos['item_snr_value']

        selected_item_snr_value = []
        for idx, snr_value in enumerate(item_snr_value):
            if snr_value in self.snr_set:
                item_filename.append(data_infos['item_filename'][idx])
                item_mod_label.append(data_infos['item_mod_label'][idx])
                selected_item_snr_value.append(snr_value)

        data_infos['item_filename'] = item_filename
        data_infos['item_mod_label'] = item_mod_label
        data_infos['item_snr_value'] = selected_item_snr_value

        snr_to_label = {snr: index for index, snr in enumerate(sorted(list(set(selected_item_snr_value))))}
        label_to_snr = {snr_to_label[snr]: snr for snr in snr_to_label}

        data_infos['snr_to_label'] = snr_to_label
        data_infos['label_to_snr'] = label_to_snr
        data_infos['snr_to_index'] = snr_to_label

        item_snr_index = []
        item_snr_label = []
        for snr_value in data_infos['item_snr_value']:
            item_snr_index.append(snr_to_label[snr_value])
            item_snr_label.append(snr_to_label[snr_value])

        data_infos['item_snr_index'] = item_snr_index
        data_infos['item_snr_label'] = item_snr_label

        return data_infos


@PREPROCESSES.register_module()
class SampleByRatio:
    def __init__(self, sample_ratio):
        self.sample_ratio = sample_ratio

    def __call__(self, data_infos):
        item_filename = data_infos['item_filename']
        item_mod_label = data_infos['item_mod_label']
        item_snr_label = data_infos['item_snr_label']
        item_snr_index = data_infos['item_snr_index']
        item_snr_value = data_infos['item_snr_value']

        left = 0
        right = -1
        cur_snr = item_snr_value[0]
        new_item_filename = []
        new_item_mod_label = []
        new_item_snr_label = []
        new_item_snr_index = []
        new_item_snr_value = []
        item_snr_value.append(None)
        for idx, snr_value in enumerate(item_snr_value):
            right += 1
            if cur_snr != snr_value:
                cur_snr_num = right - left
                save_num = math.ceil(cur_snr_num * self.sample_ratio)

                cur_item_filename = copy.copy(item_filename[left:left + save_num])
                new_item_filename.extend(cur_item_filename)

                cur_item_mod_label = copy.copy(item_mod_label[left:left + save_num])
                new_item_mod_label.extend(cur_item_mod_label)

                cur_item_snr_label = copy.copy(item_snr_label[left:left + save_num])
                new_item_snr_label.extend(cur_item_snr_label)

                cur_item_snr_index = copy.copy(item_snr_index[left:left + save_num])
                new_item_snr_index.extend(cur_item_snr_index)

                cur_item_snr_value = copy.copy(item_snr_value[left:left + save_num])
                new_item_snr_value.extend(cur_item_snr_value)

                cur_snr = snr_value
                left = right

        data_infos['item_filename'] = new_item_filename
        data_infos['item_mod_label'] = new_item_mod_label
        data_infos['item_snr_label'] = new_item_snr_label
        data_infos['item_snr_index'] = new_item_snr_index
        data_infos['item_snr_value'] = new_item_snr_value

        return data_infos
