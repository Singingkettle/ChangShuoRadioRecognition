import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class NormalizeIQ:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.mean = self.mean.reshape(-1, 1)
        self.std = self.std.reshape(-1, 1)

    def __call__(self, results):
        results['iqs'] = (results['iqs'] - self.mean) / self.std

        return results


@PIPELINES.register_module()
class NormalizeAP:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.mean = self.mean.reshape(-1, 1)
        self.std = self.std.reshape(-1, 1)

    def __call__(self, results):
        results['aps'] = (results['aps'] - self.mean) / self.std

        return results


@PIPELINES.register_module()
class NormalizeConstellation:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['cos'] = (results['cos'] - self.mean) / self.std

        return results


@PIPELINES.register_module()
class ChannelMode:
    def __call__(self, results):
        if 'iqs' in results:
            results['iqs'] = np.reshape(results['iqs'], [2, 1, -1])
        if 'aps' in results:
            results['aps'] = np.reshape(results['aps'], [2, 1, -1])

        return results
