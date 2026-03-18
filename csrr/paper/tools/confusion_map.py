import os

from .base import BaseDraw
from ..builder import FIGURES


@FIGURES.register_module()
class ConfusionMap(BaseDraw):
    def __init__(self, dataset, legend=None, scatter=None):
        super().__init__(dataset)

    def __call__(self, performances, save_dir):
        for dataset_name in self.dataset:
            for method_name in self.dataset[dataset_name]:
                performance = performances[dataset_name][method_name]
                confusion_matrix = performance.confusion_matrix
                for snr in confusion_matrix:
                    conf = confusion_matrix[snr][:-1, :-1]
                    save_path = os.path.join(save_dir, f'ConfusionMap_{snr}_{method_name}_{dataset_name}.pdf')
                    self._draw_confusion_map(conf, snr, save_path, performance.classes)
