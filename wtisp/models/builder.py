import torch.nn as nn

from ..common.utils import Registry, build_from_cfg

BACKBONES = Registry('backbone')
HEADS = Registry('head')
LOSSES = Registry('loss')
TASKS = Registry('task')
FBS = Registry('fb')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_fb(cfg):
    """Build backbone."""
    return build(cfg, FBS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_task(cfg, train_cfg=None, test_cfg=None):
    """Build task"""
    return build(cfg, TASKS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
