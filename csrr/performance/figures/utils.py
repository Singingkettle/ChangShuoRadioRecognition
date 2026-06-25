#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: utils.py
Author: Citybuster
Time: 2021/5/31 21:45
Email: chagshuo@bupt.edu.cn
"""
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

plt.rcParams["font.family"] = "Times New Roman"


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def get_new_fig(fn, fig_size=None):
    """ Init graphics """
    if fig_size is None:
        fig_size = [9, 9]
    fig = plt.figure(fn, fig_size)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing performance_info
    return fig, ax


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate performance_info such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override performance_info so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def load_json_log(json_log):
    """Parse a json-lines mmengine scalar log.

    Each line is a json object like ``{"epoch": 1, "loss": 0.5, ...}``. Lines
    without an ``epoch`` key are skipped.
    """
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            log = json.loads(line)
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict


# --------------------------------------------------------------------------- #
# mmengine text-log fallback parser
# --------------------------------------------------------------------------- #

# Examples of lines we want to capture:
#   Epoch(train)   [1][ 50/207]  lr: 4.0000e-04  ... loss: 7.1433  ...
#   Epoch(val) [3][138/138]    accuracy/top1: 42.2636  loss/classification: 2.2047 ...

_EPOCH_LINE_RE = re.compile(
    r'Epoch\((?P<mode>train|val)\)\s*\[(?P<epoch>\d+)\]\[\s*(?P<iter>\d+)/\s*(?P<total>\d+)\]')
_METRIC_RE = re.compile(r'([A-Za-z_][\w./\-]*)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')


def parse_mmengine_text_log(log_path):
    """Parse an mmengine ``*.log`` text file into ``{epoch: {metric: [...]}}``.

    For ``Epoch(train)`` lines we collect every metric (e.g. ``loss``,
    ``loss_amc_merge``) into the corresponding epoch bucket; we keep the
    iteration order so that ``values[-1]`` is the latest seen value for that
    epoch (typically the end-of-epoch summary).

    For ``Epoch(val)`` lines we collect the same way; metrics such as
    ``accuracy/top1`` and ``loss/classification`` are the typical validation
    outputs.

    The mode string is recorded under the special key ``'mode'`` (a list with
    one entry per parsed line in that epoch) so consumers can disambiguate
    train vs val.
    """
    log_dict = dict()
    if not os.path.isfile(log_path):
        return log_dict

    with open(log_path, 'r') as f:
        for line in f:
            m = _EPOCH_LINE_RE.search(line)
            if not m:
                continue
            mode = m.group('mode')
            epoch = int(m.group('epoch'))

            payload = line[m.end():]
            metrics = {k: float(v) for k, v in _METRIC_RE.findall(payload)
                       if k != 'eta'}
            if not metrics:
                continue

            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            log_dict[epoch]['mode'].append(mode)
            for k, v in metrics.items():
                log_dict[epoch][k].append(v)
    return log_dict


def find_mmengine_logs(run_dir):
    """Locate the per-run scalar log files inside an mmengine work directory.

    Returns a list of dicts ``[{'json': path_or_None, 'log': path_or_None,
    'timestamp': str}]`` sorted by timestamp ascending. ``run_dir`` is expected
    to be the per-method work-directory (e.g.
    ``work_dirs/cnn2_iq-deepsig-201610A``).
    """
    runs = []
    if not os.path.isdir(run_dir):
        return runs
    for entry in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, entry)
        if not os.path.isdir(sub):
            continue
        json_path = os.path.join(sub, 'vis_data', f'{entry}.json')
        if not os.path.isfile(json_path):
            json_path = os.path.join(sub, 'vis_data', 'scalars.json')
            if not os.path.isfile(json_path):
                json_path = None
        log_path = os.path.join(sub, f'{entry}.log')
        if not os.path.isfile(log_path):
            log_path = None
        if json_path is None and log_path is None:
            continue
        runs.append({'json': json_path, 'log': log_path, 'timestamp': entry})
    return runs


def load_run_logs(run_dir):
    """Aggregate per-epoch metrics from all runs in ``run_dir``.

    Later runs overwrite earlier runs for the same epoch (so resumed runs land
    cleanly). Returns ``{epoch: {metric: [values]}}``.
    """
    aggregated = dict()
    for run in find_mmengine_logs(run_dir):
        per_run = None
        if run['json']:
            try:
                per_run = load_json_log(run['json'])
            except Exception:  # noqa: BLE001
                per_run = None
        if not per_run and run['log']:
            per_run = parse_mmengine_text_log(run['log'])
        if not per_run:
            continue
        for epoch, metrics in per_run.items():
            aggregated[epoch] = metrics
    return aggregated
