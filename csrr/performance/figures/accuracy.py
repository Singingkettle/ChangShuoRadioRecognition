import os

import numpy as np

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
                method_names = self.dataset[dataset_name][set_name]

                common_snrs = None
                for method_name in method_names:
                    name = method_name if isinstance(method_name, str) else method_name[0]
                    snr_set = set(performances[dataset_name][name].snr_set)
                    common_snrs = snr_set if common_snrs is None else common_snrs & snr_set
                common_snrs = sorted(common_snrs)

                methods = []
                accs = []
                for method_name in method_names:
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
                    for snr in common_snrs:
                        key = f'{snr}dB' if isinstance(snr, str) else f'{snr:d}dB'
                        method['point'].append(accuracy[key])
                    methods.append(method)

                methods = [x for _, x in sorted(
                    zip(accs, methods), key=lambda p: p[0], reverse=True)]
                save_path = os.path.join(
                    save_dir,
                    f'SNRVsAccuracy_{set_name}_{dataset_name}_plot.pdf')
                self._draw_plot(methods, self.legend, common_snrs,
                                'SNR', 'Accuracy', 'SNR Vs. Accuracy',
                                save_path)
                save_path = os.path.join(
                    save_dir,
                    f'SNRVsAccuracy_{set_name}_{dataset_name}_radar.pdf')
                self._draw_radar(
                    methods, self.legend,
                    [f'{snr}dB' for snr in common_snrs],
                    'SNR Vs. Accuracy', save_path)
