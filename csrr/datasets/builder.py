import random
from functools import partial

import numpy as np
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate

from .samplers import GroupSampler, DistributedGroupSampler, DistributedSampler
from ..common.parallel import collate
from ..common.utils import Registry, build_from_cfg
from ..runner import get_dist_info

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
PREPROCESSES = Registry('preprocess')
EVALUATES = Registry('evaluate')
MERGES = Registry('merge')
FORMATS = Registry('format')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The figure_configs of modules, is either a dict
            or a list of figure_configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        performance: A list of performance handle.
    """
    if isinstance(cfg, list):
        datas = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return datas
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_preprocess(cfg, default_args=None):
    preprocess = build(cfg, PREPROCESSES, default_args)

    return preprocess


def build_evaluate(cfg, default_args=None):
    evaluate = build(cfg, EVALUATES, default_args)

    return evaluate


def build_format(cfg, default_args=None):
    save = build(cfg, FORMATS, default_args)

    return save


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=False,
                     shuffle=True,
                     seed=None,
                     is_det=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: False.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """

    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that signals on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if is_det:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(pseudo_collate),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
