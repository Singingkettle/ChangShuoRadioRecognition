from ..builder import PIPELINES
from ...common import DataContainer as DC


def recurrence_copy_dict(data, results, keys):
    if isinstance(keys, str):
        data[keys] = results[keys]
        return data
    elif isinstance(keys, dict):
        for k, v in keys.items():
            data[k] = dict()
            return recurrence_copy_dict(data[k], results[k], v)
    elif isinstance(keys, list) or isinstance(keys, tuple):
        for k in keys:
            recurrence_copy_dict(data, results, k)
        return data
    else:
        raise TypeError(f'Unsupported keys type {type(keys)}')


@PIPELINES.register_module()
class Collect:

    def __init__(self,
                 keys,
                 meta_keys=('file_name',)):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
        """

        data = dict()
        input_metas = dict()
        for key in self.meta_keys:
            if key in results:
                input_metas[key] = results[key]
            else:
                input_metas[key] = 0
        data['input_metas'] = DC(input_metas, cpu_only=True)

        data = recurrence_copy_dict(data, results, self.keys)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
