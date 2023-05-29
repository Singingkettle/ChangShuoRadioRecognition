import os

from .base import BaseDraw
from ..builder import FIGURES


@FIGURES.register_module()
class SNRVsAccuracy(BaseDraw):
    def __init__(self, dataset, legend=None, scatter=None):
        super().__init__(dataset)
        self.legend = legend
        self.scatter = scatter
        self.xticklabel_rotation = 50

    def __call__(self, performances, save_dir):
        for dataset_name in self.dataset:
            for set_name in self.dataset[dataset_name]:
                methods = []
                accs = []
                SNRS = None
                for method_name in self.dataset[dataset_name][set_name]:
                    if isinstance(method_name, str):
                        performance = performances[dataset_name][method_name]
                        method = dict(score=0, point=[], name=method_name)
                    else:
                        performance = performances[dataset_name][method_name[0]]
                        method = dict(score=0, point=[], name=method_name[1])
                        self.legend[method_name[1]] = self.legend[method_name[0]]
                    accuracy = performance.ACC
                    method['score'] = accuracy['All SNRs']
                    accs.append(method['score'])
                    SNRS = performance.snr_set
                    for snr in performance.snr_set:
                        method['point'].append(accuracy[f'{snr:d}dB'])
                    methods.append(method)
                methods = [x for _, x in sorted(zip(accs, methods), key=lambda pair: pair[0], reverse=True)]
                SNRS = [f'{snr:d}' for snr in SNRS]
                save_path = os.path.join(save_dir, f'SNRVsAccuracy_{set_name}_{dataset_name}_plot.pdf')
                self._draw_plot(methods, self.legend, SNRS, 'SNR', 'Accuracy', 'SNR Vs. Accuracy', save_path)
                save_path = os.path.join(save_dir, f'SNRVsAccuracy_{set_name}_{dataset_name}_radar.pdf')
                self._draw_radar(methods, self.legend, [f'{snr}dB' for snr in SNRS], 'SNR Vs. Accuracy',
                                 save_path)
