from ..builder import AUGMENTS


@AUGMENTS.register_module()
class FilterBySNR:
    def __init__(self, low_snr=None, high_snr=None):
        self.low_snr = low_snr
        self.high_snr = high_snr

    def __call__(self, data_infos):
        item_filename = []
        item_mod_label = []
        item_snr_value = data_infos['item_snr_value']

        selected_item_snr_value = []
        for idx, snr_value in enumerate(item_snr_value):
            if self.low_snr is not None:
                if snr_value < self.low_snr:
                    continue
            if self.high_snr is not None:
                if snr_value > self.high_snr:
                    continue
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
