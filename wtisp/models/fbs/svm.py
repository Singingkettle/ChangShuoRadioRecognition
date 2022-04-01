import joblib
import numpy as np
from sklearn import svm
from tqdm import tqdm

from ..builder import FBS


@FBS.register_module()
class SVM(object):
    def __init__(self, regularization, max_iter, model_path, method_name, mode='train'):
        super(SVM, self).__init__()
        self.model_path = model_path
        self.method_name = method_name
        self.mode = mode
        self.svm = svm.SVC(C=regularization, tol=0.005, verbose=3, max_iter=max_iter, decision_function_shape='ovo',
                           shrinking=0)
        if mode is 'test':
            self.load_model()

    def load_model(self):
        self.svm = joblib.load(self.model_path)

    def save_model(self):
        joblib.dump(self.svm, self.model_path)

    def __call__(self, data, label=None):
        if self.mode is 'train':
            self.svm.fit(data, label)
            self.save_model()
            return True
        else:
            label = []
            for item in tqdm(data):
                item = np.reshape(item, (1, -1))
                label.append({'Final': self.svm.predict(item)})
            return label
