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
            results['inputs']['iqs'] = np.reshape(results['inputs']['iqs'], [2, 1, -1])
        if 'aps' in results['inputs']:
            results['inputs']['aps'] = np.reshape(results['inputs']['aps'], [2, 1, -1])

        return results


@PIPELINES.register_module()
class TRNetProcessIQ:
    def __init__(self, M=16, is_cache=True):
        self.M = M
        self.is_cache = is_cache
        if is_cache:
            self._cache = dict()
        else:
            self._cache = None

    def __call__(self, results):
        if self.is_cache and results['file_name'] in self._cache:
            results['inputs']['imgs'] = self._cache[results['file_name']]
        else:
            x = np.transpose(np.reshape(np.squeeze(results['inputs']['iqs']), [2, -1, self.M]), [1, 0, 2])
            x = np.reshape(x, [-1, self.M])
            x = np.expand_dims(x, axis=0)
            x = np.ascontiguousarray(x)
            results['inputs']['imgs'] = x
            if self.is_cache:
                self._cache[results['file_name']] = x
        return results
