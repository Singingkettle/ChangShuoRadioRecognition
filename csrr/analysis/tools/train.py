from .base import BaseDraw
from ..builder import FIGURES


@FIGURES.register_module()
class TrainPlot(BaseDraw):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __call__(self, performances, save_dir):
        pass
