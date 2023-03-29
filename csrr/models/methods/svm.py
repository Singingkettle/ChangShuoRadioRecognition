import joblib
import numpy as np
from sklearn import svm
from tqdm import tqdm

from ..builder import METHODS


@METHODS.register_module()
class SVM:
    def __init__(self, regularization, max_iter, model_path, mode='train'):
        super(SVM, self).__init__()
        self.model_path = model_path
        self.method_name = 'SVM'
        self.mode = mode
        self.svm = svm.SVC(C=regularization, tol=0.005, verbose=3, max_iter=max_iter,
                           decision_function_shape='ovo', probability=True,
                           shrinking=0)
        if mode == 'test':
            self.load_model()

    def load_model(self):
        self.svm = joblib.load(self.model_path)

    def save_model(self):
        joblib.dump(self.svm, self.model_path)

    def __call__(self, data, label=None):
        if self.mode == 'train':
            self.svm.fit(data, label)
            self.save_model()
            return True
        else:
            inputs = []
            for item in tqdm(data):
                item = np.reshape(item, (1, -1))
                inputs.append(item)
            inputs = np.concatenate(inputs, axis=0)
            # pres = self.svm.predict_proba(inputs)
            ps = self.svm.predict(inputs)
            pres = np.zeros((ps.size, ps.max() + 1))
            pres[np.arange(ps.size), ps] = 1
            return pres
