from typing import Union

import numpy.linalg as LA

from csrr.registry import TRANSFORMS
from .base import BaseTransform

Number = Union[int, float]


@TRANSFORMS.register_module()
class SelfNormalize(BaseTransform):
    """SelfNormalize the tensor.

    Args:
        norms (Dict(str, Dict)): Configs to make selfnormalize about input numpy array

            Every value in the norms is a dict, which includes:
                ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional Order of the norm (see table under ``Notes``).
                inf means numpy's `inf` object. The default is None.
                axis : {None, int, 2-tuple of ints}, optional.
                    If `axis` is an integer, it specifies the axis of `x` along which to
                    compute the vector norms.  If `axis` is a 2-tuple, it specifies the
                    axes that hold 2-D matrices, and the matrix norms of these matrices
                    are computed.  If `axis` is None then either a vector norm (when `x`
                    is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
                    is None.
                keep_dims : bool, optional
                If this is set to True, the axes which are normed over are left in the
                result as dimensions with size one.  With this option the result will
                broadcast correctly against the original `x`.
    """

    def __init__(self, norms) -> None:
        self.norms = norms

    def transform(self, results: dict) -> dict:
        """Function to normalize tensors.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, key f'{data}_norm_cfg' key is added in to
            result dict.
        """

        def norm(x, cfg):
            x = x / LA.norm(x, **cfg)
            return x

        for key in self.norms:
            results[key] = norm(results[key], self.norms[key])
            results[f'{key}_norm_cfg'] = self.norms[key]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(norms={self.norms})'
        return repr_str
