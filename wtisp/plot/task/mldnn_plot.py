import os
import shutil

from .legend_config import LegendConfig
from ..builder import PLOTS, build_confusion_map, build_train_test_curve, build_snr_modulation, build_summary


@PLOTS.register_module()
class MLDNNPlot(object):
    def __init__(self, log_dir, config, legend, confusion_map=None,
                 train_test_curve=None, snr_modulation=None,
                 summary=None, clean_old_version=True,
                 config_legend_map=None, config_method_map=None):
        self.log_dir = log_dir
        self.config = config
        self.legend_config = LegendConfig(len(legend))
        self.save_dir = os.path.join(self.log_dir, self.config, 'fig')
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if clean_old_version:
                shutil.rmtree(self.save_dir)
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

        self.plot_blocks = list()
        if confusion_map is not None:
            if isinstance(confusion_map, dict):
                self.plot_blocks.append(build_confusion_map(confusion_map))
            elif isinstance(confusion_map, list):
                self.plot_blocks.extend(build_confusion_map(confusion_map))
            else:
                raise ValueError('The confusion maps must be list or dict!')

        if train_test_curve is not None:
            if isinstance(train_test_curve, dict):
                self.plot_blocks.append(build_train_test_curve(
                    train_test_curve, legend_config=self.legend_config))
            elif isinstance(train_test_curve, list):
                self.plot_blocks.extend(build_train_test_curve(
                    train_test_curve, legend_config=self.legend_config))
            else:
                raise ValueError('The confusion maps must be list or dict!')

        if snr_modulation is not None:
            if isinstance(snr_modulation, dict):
                self.plot_blocks.append(build_snr_modulation(
                    snr_modulation, legend_config=self.legend_config))
            elif isinstance(snr_modulation, list):
                self.plot_blocks.extend(build_snr_modulation(
                    snr_modulation, legend_config=self.legend_config))
            else:
                raise ValueError('The confusion maps must be list or dict!')

        self.plot_blocks.append(
            build_summary(summary, config_legend_map=config_legend_map, config_method_map=config_method_map))

    def plot(self, ):
        if self.plot_blocks is not None:
            for plot_block in self.plot_blocks:
                plot_block.plot(self.save_dir)
