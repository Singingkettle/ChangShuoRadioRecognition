import joblib
from sklearn import tree

from ..builder import FBS


@FBS.register_module()
class Tree(object):
    def __init__(self, model_path, max_depth=2, mode='train'):
        super(Tree, self).__init__()
        self.model_path = model_path
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
            label = self.tree.predict(data)
            return label
        else:
            return None
