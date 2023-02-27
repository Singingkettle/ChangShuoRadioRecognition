from csrr.common.utils import Registry, build_from_cfg

FIGURES = Registry('figure')
TABLES = Registry('table')
PERFORMANCES = Registry('performance')


def build(cfg, registry, default_args=None):
    """Build a performance.

    Args:
        cfg (dict, list[dict]): The figure_configs of modules, is is either a dict
            or a list of figure_configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        performance: A list of performance handle.
    """
    if isinstance(cfg, list):
        performances = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return performances
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_table(cfg, **kwargs):
    """Build confusion map."""
    return build(cfg, TABLES, kwargs)


def build_figure(cfg, **kwargs):
    """Build figure."""

    return build(cfg, FIGURES, kwargs)


def build_performance(cfg, **kwargs):
    """Build performance"""
    return build(cfg, PERFORMANCES, kwargs)
