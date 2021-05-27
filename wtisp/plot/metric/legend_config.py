import random


class LegendConfig:
    def __init__(self, num_methods):
        self.colors = [
            'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
            'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet',
            'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
            'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta',
            'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
            'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen',
            'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey',
            'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
            'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
            'lightgrey',
            'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
            'lightsteelblue', 'lightyellow',
            'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
            'mediumpurple', 'mediumseagreen',
            'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream',
            'mistyrose', 'moccasin', 'navajowhite',
            'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen',
            'paleturquoise', 'palevioletred',
            'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red',
            'rosybrown', 'royalblue', 'saddlebrown',
            'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray',
            'slategrey', 'snow', 'springgreen',
            'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
            'yellow', 'yellowgreen'
        ]
        random.Random(3).shuffle(self.colors)

        # linestyles = ['-', '--', '-.', ':'] * 6
        self.linestyles = ['-'] * num_methods
        self.markers = [
            '.', 'o', 'v', '^', '<', '>',
            '1', '2', '3', '4', '8', 'P', 'p',
            '*', 'h', 'H', '+', 'x', 'X', 's', 'd'
        ]
        random.Random(4).shuffle(self.markers)

        self.legend_configs = list()
        for i in range(num_methods):
            legend_config = dict(
                color=self.colors[i], linestyle=self.linestyles[i], marker=self.markers[i])
            self.legend_configs.append(legend_config)

    def __getitem__(self, i):
        return self.legend_configs[i]
