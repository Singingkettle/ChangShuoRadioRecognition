import os
import copy
from .base import BaseDraw
from ..builder import FIGURES


def reorder_f1score(methods, f1s, classes):
    if len(methods) == 1:
        min_class_scores = methods[0]['point']
    else:
        num_method = len(methods)
        min_class_scores = copy.deepcopy(methods[0]['point'])
        for class_name in classes:
            for method_index in range(1, num_method):
                if min_class_scores[class_name] > methods[method_index]['point'][class_name]:
                    min_class_scores[class_name] = methods[method_index]['point'][class_name]
    min_class_scores = [min_class_scores[class_name] for class_name in classes]
    classes = [x for _, x in sorted(zip(min_class_scores, classes), key=lambda pair: pair[0], reverse=True)]
    for method in methods:
        method['point'] = [method['point'][class_name] for class_name in classes]
    methods = [x for _, x in sorted(zip(f1s, methods), key=lambda pair: pair[0], reverse=True)]

    return methods, classes


@FIGURES.register_module()
class ClassVsF1ScoreWithSNR(BaseDraw):
    def __init__(self, dataset, plot_config=None, legend=None, scatter=None):
        super().__init__(dataset, plot_config)
        if plot_config is None:
            self.plot_config = dict(loc='lower left', prop={'size': 14, 'weight': 'bold'})
        else:
            self.plot_config = plot_config
        self.legend = legend
        self.scatter = scatter
        self.xticklabel_rotation = 25

    def __call__(self, performances, save_dir):
        for dataset_name in self.dataset:
            for set_name in self.dataset[dataset_name]:
                methods = dict()
                f1s = dict()
                classes = []
                for method_name in self.dataset[dataset_name][set_name]:
                    if isinstance(method_name, str):
                        performance = performances[dataset_name][method_name]
                    else:
                        performance = performances[dataset_name][method_name[0]]
                        self.legend[method_name[1]] = self.legend[method_name[0]]
                    f1_score = performance.F1
                    for group_name in f1_score:
                        if group_name == 'All SNRs':
                            if isinstance(method_name, str):
                                method = dict(score=f1_score[group_name]['Mean'], point=dict(), name=method_name)
                            else:
                                method = dict(score=f1_score[group_name]['Mean'], point=dict(), name=method_name[1])
                            classes = performance.classes
                            for class_name in classes:
                                method['point'][class_name] = f1_score[group_name][class_name]
                            if group_name in methods:
                                methods[group_name].append(method)
                                f1s[group_name].append(f1_score[group_name]['Mean'])
                            else:
                                methods[group_name] = [method]
                                f1s[group_name] = [f1_score[group_name]['Mean']]

                for group_name in methods:
                    data, xs = reorder_f1score(methods[group_name], f1s[group_name], classes)
                    save_path = os.path.join(save_dir,
                                             f'ClassVsF1Score_{group_name}_{set_name}_{dataset_name}_plot.pdf')
                    self._draw_plot(data, self.legend, xs, 'Modulation', 'F1 Score',
                                    f'Modulation Vs. F1 Score of {group_name}', save_path)
                    save_path = os.path.join(save_dir,
                                             f'ClassVsF1Score_{group_name}_{set_name}_{dataset_name}_radar.pdf')
                    self._draw_radar(data, self.legend, xs, f'Modulation Vs. F1 Score of {group_name}', save_path)
