from typing import Optional

import numpy as np

from csrr.registry import TRANSFORMS
from .base import BaseTransform


@TRANSFORMS.register_module()
class LoadIQFromFile(BaseTransform):
    def __init__(self,
                 is_squeeze: bool = False,
                 to_float32: bool = False,
                 ignore_empty: bool = False,
                 ) -> None:

        self.is_squeeze = is_squeeze
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

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)

        results['iq'] = iq
        results['iq_length'] = max(iq.shape)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'is_squeeze={self.ignore_empty}, '
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32},  )')
        return repr_str


@TRANSFORMS.register_module()
class LoadAPFromFile(BaseTransform):
    def __init__(self,
                 is_squeeze: bool = False,
                 to_float32: bool = False,
                 ignore_empty: bool = False
                 ) -> None:

        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load ap frame.

         Args:
             results (dict): Result dict from
                 :class:`mmengine.dataset.BaseDataset`.

         Returns:
             dict: The dict contains loaded ap frame and meta information.
         """
        file_path = results['ap_path']

        try:
            ap = np.load(file_path)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the ap as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)

        results['ap'] = ap
        results['ap_length'] = max(ap.shape)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'is_squeeze={self.ignore_empty}, '
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32},  )')
        return repr_str


@TRANSFORMS.register_module()
class LoadConstellationFromFile(BaseTransform):
    def __init__(self,
                 to_float32: bool = False,
                 ignore_empty: bool = False
                 ) -> None:

        self.to_float32 = to_float32
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load constellation matrix.

         Args:
             results (dict): Result dict from
                 :class:`mmengine.dataset.BaseDataset`.

         Returns:
             dict: The dict contains loaded constellation matrix and meta information.
         """
        file_path = results['co_path']

        try:
            co = np.load(file_path)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if self.to_float32:
            co = co.astype(np.float32)

        co = np.expand_dims(co, axis=0)
        results['co'] = co
        results['co_shape'] = co.shape[1:]

        return results


@TRANSFORMS.register_module()
class LoadFFTromIQ(BaseTransform):
    def __init__(self,
                 is_squeeze: bool = False,
                 to_float32: bool = False,
                 ignore_empty: bool = False,
                 ) -> None:

        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.ignore_empty = ignore_empty

    def __call__(self, results):
        try:
            iq = results['iq']
            iq = np.squeeze(iq)
            iq = iq[0, :] + 1j * iq[1, :]
            ft = np.fft.fft(iq)
            ft = np.fft.fftshift(ft)
            amplitude = np.abs(ft)
            phase = np.angle(ft)
            fft_iq = np.vstack((amplitude, phase))
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            fft_iq = fft_iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            fft_iq = np.expand_dims(ft, axis=0)

        results['fft_iq'] = fft_iq
        results['fft_iq_length'] = max(fft_iq.shape)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'is_squeeze={self.ignore_empty}, '
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32},  )')
        return repr_str
