import joblib
import numpy as np
from sklearn import tree
from tqdm import tqdm

from ..builder import FBS


@FBS.register_module()
class Tree(object):
    def __init__(self, model_path, method_name, max_depth=2, mode='train'):
        super(Tree, self).__init__()
        self.model_path = model_path
        self.method_name = method_name
        self.mode = mode
        if mode is 'train':
            self.tree = tree.DecisionTreeClassifier(max_depth=max_depth)
        elif mode is 'test':
            self.load_model()
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

    def load_model(self):
        self.tree = joblib.load(self.model_path)

    def save_model(self):
        joblib.dump(self.tree, self.model_path)

    def __call__(self, data, label=None):
        if self.mode is 'train':
            self.tree.fit(data, label)
            self.save_model()
            return True
        elif self.mode is 'test':
            label = []
            for item in tqdm(data):
                item = np.reshape(item, (1, -1))
                label.append({'Final': self.tree.predict(item)})
            return label
        else:
            return None
