import os
import shutil

from .legend_config import LegendConfig
from ..builder import PLOTS, build_confusion_map, build_train_test_curve, build_snr_modulation


@PLOTS.register_module()
class MLDNNPlot(object):
    def __init__(self, log_dir, config, legends, confusion_maps=None, train_test_curves=None, snr_modulation=None,
                 clean_old_version=True):
        self.log_dir = log_dir
        self.config = config
        self.legend_configs = LegendConfig(len(legends))
        self.save_dir = os.path.join(self.log_dir, self.config, 'fig')
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if clean_old_version:
                shutil.rmtree(self.save_dir)
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

        self.plot_blocks = list()
        if confusion_maps is not None:
            if isinstance(confusion_maps, dict):
                self.plot_blocks.append(build_confusion_map(confusion_maps))
            elif isinstance(confusion_maps, list):
                self.plot_blocks.extend(build_confusion_map(confusion_maps))
            else:
                raise ValueError('The confusion maps must be list or dict!')

        if train_test_curves is not None:
            if isinstance(train_test_curves, dict):
                self.plot_blocks.append(build_train_test_curve(
                    train_test_curves, legend_configs=self.legend_configs))
            elif isinstance(train_test_curves, list):
                self.plot_blocks.extend(build_train_test_curve(
                    train_test_curves, legend_configs=self.legend_configs))
            else:
                raise ValueError('The confusion maps must be list or dict!')

        if snr_modulation is not None:
            if isinstance(snr_modulation, dict):
                self.plot_blocks.append(build_snr_modulation(
                    snr_modulation, legend_configs=self.legend_configs))
            elif isinstance(snr_modulation, list):
                self.plot_blocks.extend(build_snr_modulation(
                    snr_modulation, legend_configs=self.legend_configs))
            else:
                raise ValueError('The confusion maps must be list or dict!')

    def plot(self, ):
        if self.plot_blocks is not None:
            for plot_block in self.plot_blocks:
                plot_block.plot(self.save_dir)
