import os

from .base import BaseDraw
from ..builder import FIGURES


@FIGURES.register_module()
class ConfusionMap(BaseDraw):
    def __init__(self, dataset, has_SNR=True, legend_config=None, scatter_config=None):
        super().__init__(dataset)
        self.has_SNR = has_SNR

    def __call__(self, performances, save_dir):
        for dataset_name in self.dataset:
            for method_name in self.dataset[dataset_name]:
                performance = performances[dataset_name][method_name]
                confusion_matrix = performance.confusion_matrix
                if self.has_SNR:
                    for snr in performance.snr_to_index:
                        conf = confusion_matrix[f'{snr:02d}dB']
                        save_path = os.path.join(save_dir, f'ConfusionMap_{snr:02d}dB_{method_name}_{dataset_name}.pdf')
                        self._draw_confusion_map(conf, save_path, performance.classes)
