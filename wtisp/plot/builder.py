from wtisp.common.utils import Registry, build_from_cfg

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


def build_flops(cfg):
    """Build flops"""
    return build(cfg, FLOPS)


def build_vis_features(cfg, scatter_config=None):
    """Build vis fea."""
    return build(cfg, VISFEATURES, dict(scatter_config=scatter_config))


def build_summary(cfg, config_legend_map=None, config_method_map=None):
    """Build summary"""
    return build(cfg, SUMMARIES, dict(config_legend_map=config_legend_map, config_method_map=config_method_map))


def build_confusion_map(cfg):
    """Build confusion map."""
    return build(cfg, CONFUSIONS)


def build_loss_accuracy_plot(cfg, legend_config=None):
    """Build train test curve."""
    return build(cfg, LOSSACCURACIES, dict(legend_config=legend_config))


def build_accuracy_f1_plot(cfg, legend_config=None):
    """Build snr accuracy."""
    return build(cfg, ACCURACYF1S, dict(legend_config=legend_config))


def build_plot(cfg, config_legend_map=None, config_method_map=None):
    """Build plot"""
    return build(cfg, PLOTS, dict(config_legend_map=config_legend_map, config_method_map=config_method_map))
