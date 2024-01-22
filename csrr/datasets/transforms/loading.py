from typing import Optional

import numpy as np

from csrr.registry import TRANSFORMS
from .base import BaseTransform


@TRANSFORMS.register_module()
class LoadIQFromFile(BaseTransform):
    def __init__(self,
                 to_float32: bool = True,
                 ignore_empty: bool = False,
                 ) -> None:

        self.to_float32 = to_float32
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load iq frame.

         Args:
             results (dict): Result dict from
                 :class:`mmengine.dataset.BaseDataset`.

         Returns:
             dict: The dict contains loaded iq frame and meta information.
         """
        file_path = results['iq_path']

        try:
            iq = np.load(file_path)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if self.to_float32:
            iq = iq.astype(np.float32)

        results['iq'] = iq
        results['iq_length'] = max(iq.shape)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32},  )')
        return repr_str


