from .base import BasePerformance
from ..builder import PERFORMANCES


@PERFORMANCES.register_module()
class Classification(BasePerformance):
    def __init__(self, publish, res_dir, figure=None, table=None):
        super().__init__()


