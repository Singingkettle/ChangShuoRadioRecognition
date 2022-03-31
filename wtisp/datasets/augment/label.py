from ..builder import AUGMENTS


@AUGMENTS.register_module()
class MLDNNSNRLabel:
    def __init__(self, snr_threshold=0, item_weights=None):
        self.snr_threshold = snr_threshold
        self.item_weights = item_weights

    def __call__(self, data_infos):
        item_snr_value = data_infos['item_snr_value']
        low_weights = []
        high_weights = []
        item_snr_label = []
        for item_index, item in enumerate(item_snr_value):
            if item >= self.snr_threshold:
                item_snr_label.append(0)
                if self.item_weights is not None:
                    low_weights.append(self.item_weights[0])
                    high_weights.append(self.item_weights[1])
            else:
                item_snr_label.append(1)
                if self.item_weights is not None:
                    low_weights.append(self.item_weights[1])
                    high_weights.append(self.item_weights[0])

        if self.item_weights is not None:
            data_infos['low_weights'] = low_weights
            data_infos['high_weights'] = high_weights

        snr_to_label = {'SNR0': 0, 'SNR1': 1}
        label_to_snr = {0: 'SNR0', 1: 'SNR1'}
        data_infos['snr_to_label'] = snr_to_label
        data_infos['label_to_snr'] = label_to_snr
        data_infos['item_snr_label'] = item_snr_label

        return data_infos
