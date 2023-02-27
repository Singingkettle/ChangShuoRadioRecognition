from abc import ABCMeta, abstractmethod


class BasePerformance(metaclass=ABCMeta):
    """Base class for performance."""

    def __init__(self):
        super(BasePerformance, self).__init__()

    @abstractmethod
    def generate_figures(self, data):
        pass

    @abstractmethod
    def generate_tables(self, data):
        pass

    @abstractmethod
    def out(self, data):
        pass
