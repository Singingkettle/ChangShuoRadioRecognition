import os
import pickle

from ..builder import PERFORMANCES, build_figure, build_table
from ..figure_configs import generate_legend_config, generate_scatter_config
from ..metrics import ClassificationMetricsWithSNRForSingle


@PERFORMANCES.register_module()
class Classification:
    """Aggregate per-method ``paper.pkl`` files and dispatch them to figures.

    Configuration shape::

        info = dict(
            work_dir='work_dirs',
            save_dir='work_dirs/performance',
            methods={'CNN2': 0, ...},
            publish=dict(
                deepsig201610A=dict(CNN2='cnn2_iq-deepsig-201610A', ...),
            ),
        )

    For each (dataset, method) entry under ``publish`` we try to load
    ``<work_dir>/<subdir>/res/paper.pkl``. Missing pickles are warned about
    and the method is silently dropped from the in-memory ``performances``
    map so that figures can degrade gracefully.

    The original ``info`` dict is re-exposed to figure/table classes via a
    reserved key ``performances['_info']`` (this key is ignored by all the
    figure classes which iterate over their own configured ``dataset`` map).
    """

    INFO_KEY = '_info'

    def __init__(self, info, Figures=None, Tables=None):
        self.info = info
        self.work_dir = info['work_dir']
        self.save_dir = info.get('save_dir',
                                 os.path.join(self.work_dir, 'performance'))
        self.methods = info['methods']
        self.legend = generate_legend_config(self.methods)
        self.scatter = generate_scatter_config(self.methods)
        self.publish = info['publish']

        self.performances = dict()
        self.missing = []
        for dataset_name in self.publish:
            self.performances[dataset_name] = dict()
            for method in self.publish[dataset_name]:
                pkl_path = os.path.join(
                    self.work_dir, self.publish[dataset_name][method],
                    'res', 'paper.pkl')
                if not os.path.isfile(pkl_path):
                    self.missing.append((dataset_name, method, pkl_path))
                    print(f'[Classification] paper.pkl missing for '
                          f'{dataset_name}/{method}: {pkl_path}; '
                          f'method will be skipped in figures.')
                    continue
                try:
                    with open(pkl_path, 'rb') as f:
                        res = pickle.load(f)
                except Exception as exc:  # noqa: BLE001
                    self.missing.append((dataset_name, method, pkl_path))
                    print(f'[Classification] failed to load {pkl_path}: '
                          f'{exc}; skipping {dataset_name}/{method}.')
                    continue
                self.performances[dataset_name][method] = \
                    ClassificationMetricsWithSNRForSingle(
                        res['pps'], res['gts'], res['snrs'], res['classes'],
                        feas=res.get('feas'), centers=res.get('centers'))

        self.draw_handles = []
        if Figures is not None:
            for figure in Figures:
                self.draw_handles.append(
                    build_figure(figure, legend=self.legend,
                                 scatter=self.scatter))
        if Tables is not None:
            for table in Tables:
                self.draw_handles.append(
                    build_table(table, legend=self.legend,
                                scatter=self.scatter))

    def draw(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # Attach a transient ``_info`` entry so figure/table classes that need
        # the publish/work_dir mapping (Flops, TrainPlot, ...) can find it
        # without breaking the existing signatures of the simpler figures.
        payload = dict(self.performances)
        payload[self.INFO_KEY] = self.info
        for draw in self.draw_handles:
            try:
                draw(payload, self.save_dir)
            except Exception as exc:  # noqa: BLE001
                print(f'[Classification] handle '
                      f'{type(draw).__name__} failed: {exc}')
