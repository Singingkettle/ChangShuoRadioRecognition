import random

import matplotlib.pyplot as plt
import numpy as np


class LegendConfig:
    def __init__(self, num_methods):
        self.colors = [
            'black', 'red', 'saddlebrown', 'orange', 'green', 'darkslategray',
            'dodgerblue', 'blue', 'purple', 'pink', 'cyan', 'brown', 'olive', 'springgreen',
            'indigo', 'deeppink', 'navy', 'sienna', 'yellow', 'gray', 'darkgreen', 'mediumslateblue',
            'cornflowerblue', 'darkolivegreen', 'steelblue', 'darkviolet', 'darkgoldenrod', 'cadetblue',
            'grey', 'indianred', 'tomato', 'palegreen', 'seagreen', 'deepskyblue', 'slategray', 'royalblue',
            'magenta', 'palevioletred', 'gold', 'fuchsia', 'lawngreen', 'blueviolet', 'peru', 'peachpuff',
            'rosybrown', 'darkkhaki', 'tan'
        ]
        random.Random(10).shuffle(self.colors)

        self.linestyles = ['-', '--', '-.', ':'] * num_methods
        self.markers = [
                           '.', 'o', 'v', '^', '<', '>',
                           '1', '2', '3', '4', '8', 'P', 'p',
                           '*', 'h', 'H', '+', 'x', 'X', 's', 'd'
                       ] * num_methods

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
