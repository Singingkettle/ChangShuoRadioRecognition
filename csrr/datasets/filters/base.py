from abc import ABCMeta, abstractmethod
from typing import Dict, List


class BaseFilter(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self, data_list: List, meta_info) -> List:
        return self.filter(data_list, meta_info)

    @abstractmethod
    def filter(self, data_list: List, meta_info: Dict) -> (List, Dict):
        """The filter function. All subclass of BaseFilter should
        override this method.

        This function takes the data_list and dataset's meta_info dict as the input,
        and filter the data list in a specific rule.

        Args:
            data_list (list): The data list of dataset.
            meta_info (dict): The meta info of dataset
        Returns:
            list: The filtered data list.
            dict: New meta info based on the filtered data list
        """

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
