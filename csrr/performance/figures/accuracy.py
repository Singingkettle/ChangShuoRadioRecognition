import os

from .base import BaseDraw
from ..builder import FIGURES


@FIGURES.register_module()
class SNRVsAccuracy(BaseDraw):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)

    def __call__(self, performances, save_dir, legend_config):
        for dataset_name in self.dataset:
            for set_name in self.dataset[dataset_name]:
                methods = []
                accs = []
                SNRS = None
                for method_name in self.dataset[dataset_name][set_name]:
                    performance = performances[dataset_name][method_name]
                    accuracy = performance.accuracy
                    method = dict(score=accuracy['ACC'], point=[], name=method_name)
                    accs.append([method['ACC']])
                    SNRS = accuracy.snr_set
                    for snr in accuracy.snr_set:
                        method['point'].append(accuracy[f'{snr:02d}dB'])
                    methods.append(method)
                methods = [x for _, x in sorted(zip(accs, methods), key=lambda pair: pair[0])]
                save_path = os.path.join(save_dir, f'SNRVsAccuracy_{set_name}_{dataset_name}_plot.pdf')
                self._draw_plot(methods, legend_config, SNRS, 'SNR', 'Accuracy', 'SNR Vs. Accuracy', save_path)
                save_path = os.path.join(save_dir, f'SNRVsAccuracy_{set_name}_{dataset_name}_radar.pdf')
                self._draw_radar(methods, legend_config, [f'{snr:02d}dB' for snr in SNRS], 'SNR Vs. Accuracy',
                                 save_path)
