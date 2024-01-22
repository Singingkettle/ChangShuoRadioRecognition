from collections import defaultdict
from collections.abc import Sequence
from typing import Dict, List, Union

import numpy as np
import torch
from mmengine.utils import is_str

from csrr.registry import TRANSFORMS
from csrr.structures import DataSample, MultiTaskDataSample
from .base import BaseTransform


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the data of inputs.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~csrr.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (Union[str, List[str]]): The key of element to feed into the model forwarding.
        algorithm_keys (List[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (List[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`PackInputs.DEFAULT_META_KEYS`.

    .  admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .  admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
    """

    DEFAULT_META_KEYS = ('sample_idx',)

    def __init__(self,
                 input_key: Union[str, List[str]],
                 algorithm_keys: List[str] = (),
                 meta_keys=DEFAULT_META_KEYS) -> None:
        if isinstance(input_key, Sequence):
            if len(input_key) == 1:
                input_key = input_key[0]
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            input_ = to_tensor(input_).contiguous()
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if isinstance(self.input_key, List):
            packed_results['inputs'] = dict()
            for input_key in self.input_key:
                input_ = results[input_key]
                packed_results['inputs'][input_key] = self.format_input(input_)
        else:
            if self.input_key in results:
                input_ = results[self.input_key]
                packed_results['inputs'] = self.format_input(input_)

        data_sample = DataSample()

        # Set default keys
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        if 'mask' in results:
            data_sample.set_mask(results['mask'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f'algorithm_keys={self.algorithm_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackMultiTaskInputs(BaseTransform):
    """Convert all image labels of multitask dataset to a dict of tensor.

    Args:
        multi_task_fields (List[str]):
        input_key (Union[str, List[str]]):
        task_handlers (dict):
    """

    def __init__(self,
                 multi_task_fields: List[str],
                 input_key: Union[str, List[str]],
                 task_handlers: Dict = dict()):
        self.multi_task_fields = multi_task_fields
        if isinstance(input_key, Sequence):
            if len(input_key) == 1:
                input_key = input_key[0]
        self.input_key = input_key
        self.task_handlers = defaultdict(PackInputs)
        for task_name, task_handler in task_handlers.items():
            self.task_handlers[task_name] = TRANSFORMS.build(task_handler)

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        result = {'iq_path': 'a.npy', 'gt_label': {'task1': 1, 'task3': 3},
            'iq': array([[[  0,   0,   0]]])}
        """
        packed_results = dict()
        results = results.copy()

        if isinstance(self.input_key, Sequence):
            packed_results['inputs'] = dict()
            for input_key in self.input_key:
                input_ = results[self.input_key]
                packed_results['inputs'][input_key] = PackInputs.format_input(input_)
        else:
            packed_results['inputs'] = PackInputs.format_input([])

        task_results = defaultdict(dict)
        for field in self.multi_task_fields:
            if field in results:
                value = results.pop(field)
                for k, v in value.items():
                    task_results[k].update({field: v})

        data_sample = MultiTaskDataSample()
        for task_name, task_result in task_results.items():
            task_handler = self.task_handlers[task_name]
            task_pack_result = task_handler({**results, **task_result})
            data_sample.set_field(task_pack_result['data_samples'], task_name)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        repr = self.__class__.__name__
        task_handlers = ', '.join(
            f"'{name}': {handler.__class__.__name__}"
            for name, handler in self.task_handlers.items())
        repr += f'(multi_task_fields={self.multi_task_fields}, '
        repr += f"input_key='{self.input_key}', "
        repr += f'task_handlers={{{task_handlers}}})'
        return repr


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose numpy array.

    Args:
        orders (Dict[str, List[int]]): The output dimensions order.
    """

    def __init__(self, orders):
        self.orders = orders

    def transform(self, results):
        """Method to transpose array."""
        for key in self.orders:
            results[key] = results[key].transpose(self.orders[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(orders={self.orders})'


@TRANSFORMS.register_module()
class Reshape(BaseTransform):
    """Reshape numpy array.

    Args:
        shapes (Dict[str, List[int]]): Configs to convert numpy array.
    """

    def __init__(self, shapes):
        self.shapes = shapes

    def transform(self, results):
        """Method to transpose array."""
        for key in self.shapes:
            results[key] = results[key].reshape(self.shapes[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(shapes={self.shapes})'