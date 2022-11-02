from csrr.common.utils import Registry, build_from_cfg

CONFUSIONS = Registry('confusion')
LOSSACCURACIES = Registry('lossaccuracy')
ACCURACYF1S = Registry('accuracyf1')
SUMMARIES = Registry('summary')
VISFEATURES = Registry('fea')
FLOPS = Registry('flop')
PLOTS = Registry('plot')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        plots: A list of plot handle.
    """
    if isinstance(cfg, list):
        plots = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return plots
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_confusion_map(cfg, log_dir):
    """Build confusion map."""
    return build(cfg, CONFUSIONS, dict(log_dir=log_dir))


def build_loss_accuracy_plot(cfg, log_dir, legend, legend_config):
    """Build train test curve."""
    return build(cfg, LOSSACCURACIES, dict(log_dir=log_dir, legend=legend, legend_config=legend_config))


def build_accuracy_f1_plot(cfg, log_dir, legend, legend_config):
    """Build snr accuracy."""
    return build(cfg, ACCURACYF1S, dict(log_dir=log_dir, legend=legend, legend_config=legend_config))


def build_summary(cfg, log_dir, config_legend_map, config_method_map):
    """Build summary"""
    return build(cfg, SUMMARIES,
                 dict(log_dir=log_dir, config_legend_map=config_legend_map, config_method_map=config_method_map))


def build_vis_features(cfg, log_dir, scatter_config):
    """Build vis fea."""
    return build(cfg, VISFEATURES, dict(log_dir=log_dir, scatter_config=scatter_config))


def build_flops(cfg, log_dir):
    """Build flops"""
    return build(cfg, FLOPS, dict(log_dir=log_dir))


def build_plot(cfg, log_dir, legend, scatter, config_legend_map, config_method_map):
    """Build plot"""
    return build(cfg, PLOTS, dict(log_dir=log_dir, legend=legend, config_legend_map=config_legend_map,
                                  config_method_map=config_method_map, scatter=scatter))
