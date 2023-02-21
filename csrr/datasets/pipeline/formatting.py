from ..builder import PIPELINES
from ...common import DataContainer as DC


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

        data = {}
        input_metas = {}
        for key in self.meta_keys:
            if key in results:
                input_metas[key] = results[key]
            else:
                input_metas[key] = 0
        data['input_metas'] = DC(input_metas, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
