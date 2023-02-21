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
        results['inputs']['iqs'] = (results['inputs']['iqs'] - self.mean) / self.std

        return results


@PIPELINES.register_module()
class NormalizeAP:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.mean = self.mean.reshape(-1, 1)
        self.std = self.std.reshape(-1, 1)

    def __call__(self, results):
        results['inputs']['aps'] = (results['inputs']['aps'] - self.mean) / self.std

        return results


@PIPELINES.register_module()
class NormalizeConstellation:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['inputs']['cos'] = (results['inputs']['cos'] - self.mean) / self.std

        return results


@PIPELINES.register_module()
class ChannelMode:
    def __call__(self, results):
        if 'iqs' in results['inputs']:
            results['inputs']['iqs'] = np.reshape(results['inputs'], [2, 1, -1])
        if 'aps' in results['inputs']:
            results['inputs']['aps'] = np.reshape(results['inputs'], [2, 1, -1])

        return results
