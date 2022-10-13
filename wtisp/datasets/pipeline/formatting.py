from ..builder import PIPELINES


@PIPELINES.register_module()
class Collect:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to collect keys in results.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
        """

        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
