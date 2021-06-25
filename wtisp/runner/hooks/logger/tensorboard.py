# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

import matplotlib.pyplot as plt
import torch

from .base import LoggerHook
from ..hook import HOOKS
from ...dist_utils import master_only
from ....common.parallel import is_module_wrapper
from ....common.utils import TORCH_VERSION


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 add_graph=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir
        self.add_graph = add_graph
        self.writer = None

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.common.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if 'figure' in tag:
                fig_number = val.number
                if plt.fignum_exists(fig_number):
                    self.writer.add_figure(tag, val, self.get_step(runner))
                    plt.close(fig_number)
                else:
                    raise ValueError(
                        'The figure of confusion matrix {} is wrong. Please the code'.format(tag))
            elif isinstance(val, str):
                self.writer.add_text(tag, val, self.get_step(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_step(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()

    @master_only
    def before_epoch(self, runner):
        if runner.epoch == 0 and self.add_graph:
            if is_module_wrapper(runner.model):
                _model = runner.model.module
            else:
                _model = runner.model
            device = next(_model.parameters()).device
            data = next(iter(runner.data_loader))
            for key, value in data.items():
                data[key] = value.to(device)

            with torch.no_grad():
                self.writer.add_graph(_model, data)
