import matplotlib.pyplot as plt
import numpy as np

from .configs import _COLORS, _MARKERS, _LINES


class LegendConfig:
    def __init__(self, num_methods, seed=None):
        self.colors = _COLORS * num_methods
        self.linestyles = _MARKERS * num_methods
        self.markers = _LINES * num_methods

        self.legend_configs = list()
        for i in range(num_methods):
            legend_config = dict(
                color=self.colors[i], linestyle=self.linestyles[i], marker=self.markers[i])
            self.legend_configs.append(legend_config)

    def __getitem__(self, i):
        return self.legend_configs[i]


if __name__ == '__main__':

    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    xs = np.array(xs)
    ys = np.array(ys)
    a = LegendConfig(1)
    for i, ncolor in enumerate(a.colors):
        nys = ys + i * 10
        plt.plot(xs, nys, color=ncolor, label=ncolor)
    plt.legend()
    plt.tight_layout()  # set layout slim
    plt.savefig('color.pdf', bbox_inches='tight')
    plt.close()
