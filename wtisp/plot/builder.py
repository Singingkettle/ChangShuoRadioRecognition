from wtisp.common.utils import Registry, build_from_cfg

CONFUSIONS = Registry('confusion')
TRAINTESTCURVES = Registry('traintestcurve')
SNRMODULATIONS = Registry('snrmodulation')
SUMMARIES = Registry('summary')
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


def build_summary(cfg, config_legend_map=None, config_method_map=None):
    """Build summary"""
    return build(cfg, SUMMARIES, dict(config_legend_map=config_legend_map, config_method_map=config_method_map))


def build_confusion_map(cfg):
    """Build confusion map."""
    return build(cfg, CONFUSIONS)


def build_train_test_curve(cfg, legend_config=None):
    """Build train test curve."""
    return build(cfg, TRAINTESTCURVES, dict(legend_config=legend_config))


def build_snr_modulation(cfg, legend_config=None):
    """Build snr accuracy."""
    return build(cfg, SNRMODULATIONS, dict(legend_config=legend_config))


def build_plot(cfg, config_legend_map=None, config_method_map=None):
    """Build plot"""
    return build(cfg, PLOTS, dict(config_legend_map=config_legend_map, config_method_map=config_method_map))
