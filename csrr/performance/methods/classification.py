import os

from .base import BasePerformance
from ..builder import PERFORMANCES, build_figure, build_table
from ..figure_configs import LegendConfig, ScatterConfig
from ..method_info import AMCMethodInfo
from ..metrics import ClassificationMetricsWithSNRForSingle
from ...common.fileio import load as IOLoad


@PERFORMANCES.register_module()
class Classification(BasePerformance):
    def __init__(self, Figures=None, Tables=None):
        super().__init__()

        self.work_dir = AMCMethodInfo.work_dir
        self.methods = AMCMethodInfo.methods
        self.legend = LegendConfig(len(self.methods))
        self.scatter = ScatterConfig(len(self.methods))
        self.publish = AMCMethodInfo.publish

        self.performances = dict()
        for datasetname in self.publish:
            for method in self.publish[datasetname]:
                res = IOLoad(os.path.join(self.work_dir, self.publish[datasetname][method], 'res/paper.pkl'))
                self.performances[method] = ClassificationMetricsWithSNRForSingle(res['pts'], res['gts'], res['snrs'],
                                                                                  res['classes'], res['cfg'])

        self.draw_handles = []
        if Figures is not None:
            for figure in Figures:
                self.draw_handles.append(build_figure(figure))

        if Tables is not None:
            for table in Tables:
                self.draw_handles.append(build_table(table))


    def draw(self):
        for draw in self.draw_handles:
            draw(self.performances, self.work_dir)


