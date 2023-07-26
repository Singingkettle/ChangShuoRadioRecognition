import os

from ..builder import PERFORMANCES, build_figure, build_table
from ..figure_configs import generate_legend_config, generate_scatter_config
from ..metrics import ClassificationMetricsWithSNRForSingle
from ...common.fileio import load as IOLoad


@PERFORMANCES.register_module()
class Classification:
    def __init__(self, save_dir, info, Figures=None, Tables=None):
        self.save_dir = save_dir
        self.work_dir = info['work_dir']
        self.methods = info['methods']
        self.legend = generate_legend_config(self.methods)
        self.scatter = generate_scatter_config(self.methods)
        self.publish = info['publish']

        self.performances = dict()
        for dataset_name in self.publish:
            self.performances[dataset_name] = dict()
            for method in self.publish[dataset_name]:
                res = IOLoad(os.path.join(self.work_dir, self.publish[dataset_name][method], 'res/paper.pkl'))
                self.performances[dataset_name][method] = ClassificationMetricsWithSNRForSingle(res['pps'], res['gts'],
                                                                                                res['snrs'],
                                                                                                res['classes'],
                                                                                                feas=res.get('feas'),
                                                                                                centers=res.get(
                                                                                                    'centers'),
                                                                                                cfg=res.get('cfg'))

        self.draw_handles = []
        if Figures is not None:
            for figure in Figures:
                self.draw_handles.append(build_figure(figure, legend=self.legend, scatter=self.scatter))

        if Tables is not None:
            for table in Tables:
                self.draw_handles.append(build_table(table))


    def draw(self):
        for draw in self.draw_handles:
            draw(self.performances, self.save_dir)
