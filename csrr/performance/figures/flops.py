import os

from ..builder import TABLES


@TABLES.register_module()
class Flops:
    """Placeholder for FLOPs computation table.

    TODO: Implement using mmengine's model analysis tools.
    """
    def __init__(self, dataset, legend=None, scatter=None):
        self.dataset = dataset

    def __call__(self, performances, save_dir):
        print('[Flops] Not yet implemented in the refactored version.')
