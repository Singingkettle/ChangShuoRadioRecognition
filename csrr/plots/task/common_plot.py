import os
import shutil

from ..builder import (PLOTS, build_confusion_map, build_loss_accuracy_plot,
                       build_accuracy_f1_plot, build_summary, build_vis_features, build_flops)
from ..config.legend_config import LegendConfig
from ..config.scatter_config import ScatterConfig


@PLOTS.register_module()
class CommonPlot(object):
    def __init__(self, name, log_dir=None, legend=None, scatter=None, confusion_map=None,
                 train_test_curve=None, snr_modulation=None, summary=None, vis_fea=None,
                 flops=None, clean_old_version=True, config_legend_map=None, config_method_map=None):
        self.name = name
        self.log_dir = log_dir
        self.legend_config = LegendConfig(len(legend))
        self.scatter_config = ScatterConfig(scatter)
        self.save_dir = os.path.join(self.log_dir, self.name)
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
                self.plot_blocks.append(build_confusion_map(confusion_map, log_dir=self.log_dir))
            elif isinstance(confusion_map, list):
                self.plot_blocks.extend(build_confusion_map(confusion_map, log_dir=self.log_dir))
            else:
                raise ValueError('The confusion maps must a be list or dict!')

        if train_test_curve is not None:
            if isinstance(train_test_curve, dict):
                self.plot_blocks.append(build_loss_accuracy_plot(
                    train_test_curve, log_dir=self.log_dir, legend=self.legend, legend_config=self.legend_config))
            elif isinstance(train_test_curve, list):
                self.plot_blocks.extend(build_loss_accuracy_plot(
                    train_test_curve, log_dir=self.log_dir, legend=self.legend, legend_config=self.legend_config))
            else:
                raise ValueError('The confusion maps must be a list or dict!')

        if snr_modulation is not None:
            if isinstance(snr_modulation, dict):
                self.plot_blocks.append(build_accuracy_f1_plot(
                    snr_modulation, log_dir=self.log_dir, legend=self.legend, legend_config=self.legend_config))
            elif isinstance(snr_modulation, list):
                self.plot_blocks.extend(build_accuracy_f1_plot(
                    snr_modulation, log_dir=self.log_dir, legend=self.legend, legend_config=self.legend_config))
            else:
                raise ValueError('The confusion maps must be a list or dict!')

        if summary is not None:
            self.plot_blocks.append(
                build_summary(summary, log_dir=self.log_dir, config_legend_map=config_legend_map,
                              config_method_map=config_method_map))

        if vis_fea is not None:
            if isinstance(vis_fea, dict):
                self.plot_blocks.append(build_vis_features(vis_fea, log_dir=self.log_dir,
                                                           scatter_config=self.scatter_config))
            elif isinstance(vis_fea, list):
                self.plot_blocks.extend(build_vis_features(vis_fea, log_dir=self.log_dir,
                                                           scatter_config=self.scatter_config))
            else:
                raise ValueError('The vis features must be a list or dict!')

        if flops is not None:
            self.plot_blocks.append(build_flops(flops, log_dir=self.log_dir))

    def plot(self, ):
        if self.plot_blocks is not None:
            for plot_block in self.plot_blocks:
                plot_block.plot(self.save_dir)
