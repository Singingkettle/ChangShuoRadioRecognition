import os

from .base import BaseDraw
from ..builder import FIGURES


def reorder_f1score(methods, f1s, classes):
    if len(methods) == 1:
        min_class_scores = methods[0]['point']
    else:
        num_method = len(methods)
        min_class_scores = methods[0]['point']
        for class_name in range(classes):
            for method_index in range(1, num_method):
                if min_class_scores[class_name] > methods[method_index]['point'][class_name]:
                    min_class_scores[class_name] = methods[method_index]['point'][class_name]
    min_class_scores = [min_class_scores[class_name] for class_name in classes]
    classes = [x for _, x in sorted(zip(min_class_scores, classes), key=lambda pair: pair[0])]
    for method in methods:
        method['point'] = [method['point'][class_name] for class_name in classes]
    methods = [x for _, x in sorted(zip(f1s, methods), key=lambda pair: pair[0])]

    return methods, classes


@FIGURES.register_module()
class ClassVsF1ScorePlot(BaseDraw):
    def __init__(self, dataset, plot_config=None, has_SNR=True):
        super().__init__(dataset, plot_config)
        if plot_config is None:
            self.plot_config = dict(loc='lower left', prop={'size': 14, 'weight': 'bold'})
        else:
            self.plot_config = plot_config
        self.has_SNR = has_SNR

    def __call__(self, performances, save_dir, legend_config):
        for dataset_name in self.dataset:
            for set_name in self.dataset[dataset_name]:
                if self.has_SNR:
                    methods = dict()
                    f1s = dict()
                else:
                    methods = []
                    f1s = []
                CLASSES = None
                for method_name in self.dataset[dataset_name][set_name]:
                    performance = performances[dataset_name][method_name]
                    f1_score = performance.f1_score
                    if self.has_SNR:
                        for snr in f1_score:
                            method = dict(score=f1_score[snr]['F1'], point=[], name=method_name)
                            CLASSES = f1_score.classes
                            for class_name in CLASSES:
                                method['point'].append(f1_score[snr][class_name])
                            if snr in methods:
                                methods[snr].append(method)
                                f1s[snr].append(f1_score[snr]['F1'])
                            else:
                                methods[snr] = [method]
                                f1s[snr] = [f1_score[snr]['F1']]
                    else:
                        method = dict(score=f1_score['F1'], point=[], name=method_name)
                        CLASSES = f1_score.classes
                        for class_name in CLASSES:
                            method['point'].append(f1_score[class_name])
                        methods.append(method)

                if self.has_SNR:
                    for snr in methods:
                        data, xs = reorder_f1score(methods[snr], f1s[snr], CLASSES)
                        save_path = os.path.join(save_dir,
                                                 f'ClassVsF1Score_{snr:02d}dB_{set_name}_{dataset_name}_plot.pdf')
                        self._draw_plot(data, legend_config, xs, 'Modulation', 'F1 Score', 'Modulation Vs. F1 Score',
                                        save_path)
                        save_path = os.path.join(save_dir,
                                                 f'ClassVsF1Score_{snr:02d}dB_{set_name}_{dataset_name}_radar.pdf')
                        self._draw_radar(data, legend_config, xs, 'Modulation Vs. F1 Score', save_path)
                else:
                    save_path = os.path.join(save_dir, f'ClassVsF1Score_{set_name}_{dataset_name}_plot.pdf')
                    self._draw_plot(methods, legend_config, CLASSES, 'Modulation', 'F1 Score',
                                    'Modulation Vs. F1 Score', save_path)
                    save_path = os.path.join(save_dir, f'ClassVsF1Score_{set_name}_{dataset_name}_radar.pdf')
                    self._draw_radar(methods, legend_config, CLASSES, 'Modulation Vs. F1 Score', save_path)
