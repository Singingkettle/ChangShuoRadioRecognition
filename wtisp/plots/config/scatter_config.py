import random


class ScatterConfig:
    def __init__(self, labels):
        self.colors = [
            'black', 'red', 'saddlebrown', 'orange', 'green', 'darkslategray',
            'dodgerblue', 'blue', 'purple', 'pink', 'cyan', 'brown', 'olive', 'springgreen',
            'indigo', 'deeppink', 'navy', 'sienna', 'yellow', 'gray', 'darkgreen', 'mediumslateblue',
            'cornflowerblue', 'darkolivegreen', 'steelblue', 'darkviolet', 'darkgoldenrod', 'cadetblue'
        ]
        random.Random(3).shuffle(self.colors)
        self.markers = [
                           '.', 'o', 'v', '^', '<', '>',
                           '1', '2', '3', '4', '8', 'P', 'p',
                           '*', 'h', 'H', '+', 'x', 'X', 's', 'd'
                       ] * len(labels)

        self.scatter_config = dict()
        label_index = 0
        for label in labels:
            self.scatter_config[label] = dict(color=self.colors[label_index], marker=self.markers[label_index])
            label_index += 1

    def __getitem__(self, label):
        return self.scatter_config[label]
