from mmengine.registry import Registry

FIGURES = Registry('figure')
TABLES = Registry('table')
PERFORMANCES = Registry('performance')


def build_figure(cfg, **kwargs):
    return FIGURES.build(cfg, default_args=kwargs)


def build_table(cfg, **kwargs):
    return TABLES.build(cfg, default_args=kwargs)


def build_performance(cfg, **kwargs):
    return PERFORMANCES.build(cfg, default_args=kwargs)
