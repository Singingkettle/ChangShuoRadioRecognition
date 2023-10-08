from csrr.registry import DATA_FILTERS


@DATA_FILTERS.register_module()
class FilterBySNR:
    def __init__(self, save_snr_set):
        self.save_snr_set = save_snr_set

    def __call__(self, data_list, metainfo):
        new_data_list = []
        for data_info in data_list:
            if data_info['snr'] in self.save_snr_set:
                new_data_list.append(data_info)

        return new_data_list
