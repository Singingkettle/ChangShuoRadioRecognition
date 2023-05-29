import random

from .configs import _MARKERS, _COLORS


def generate_scatter_config(methods):
    colors = _COLORS
    random.Random(3).shuffle(colors)
    markers = _MARKERS * len(methods)

    scatter = dict()
    for index, method in enumerate(methods):
        scatter[method] = dict(color=colors[index], marker=markers[index])

    return scatter

