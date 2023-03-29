import random

from .configs import _MARKERS, _COLORS


class ScatterConfig:
    def __init__(self, labels):
        self.colors = _COLORS
        random.Random(3).shuffle(self.colors)
        self.markers = _MARKERS * len(labels)

        self.scatter_config = dict()
        label_index = 0
        for label in labels:
            self.scatter_config[label] = dict(color=self.colors[label_index], marker=self.markers[label_index])
            label_index += 1

    def __getitem__(self, label):
        return self.scatter_config[label]
