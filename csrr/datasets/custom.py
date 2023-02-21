from abc import ABCMeta, abstractmethod
import os.path as osp

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS, build_preprocess, build_evaluate, build_save
from .pipeline import Compose
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class CustomDataset(Dataset, metaclass=ABCMeta):
    """Custom dataset.
    Args:
        ann_file (str): Annotation file path.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, ``Amplitude/Phase data``,
                                   ``constellation data``.
        test_mode (bool, optional): If set True, annotation will not be loaded.

    """

    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False, preprocess=None, evaluate=None, save=None):
        self.ann_file = ann_file
        self.pipeline = pipeline
        self.data_root = data_root
        self.test_mode = test_mode
        if preprocess is not None:
            preprocess = build_preprocess(preprocess)
            for process in preprocess:
                self.data_infos = process(self.data_infos)

        if evaluate is not None:
            self.eval = build_evaluate(evaluate)
        else:
            self.eval = None

        if save is not None:
            self.save = build_save(save)
        else:
            self.save = None

        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def _set_group_flag(self):
        """Set flag according to the mean value (ms) of signal data snrs.

        set as group i, where self.SNRS[i]<=ms<self.SNRS[i+1].
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos['annotations'])

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return IOLoad(ann_file)

    @abstractmethod
    def get_ann_info(self, idx):
        """Get annotation by idx.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified idx.
        """

    def __getitem__(self, idx):

        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            data = self.prepare_train_data(idx)

        return data

    @abstractmethod
    def pre_pipeline(self, results):
        """Prepare inputs dict for pipeline."""

    def prepare_train_data(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = self.get_ann_info(idx)
        results = self.pre_pipeline(results)
        results['targets'] = dict()
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        file_name = self.data_infos['annotations'][idx]['file_name']
        results = dict(file_name=file_name)
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, out_dir, results):
        """ Format results and format.
        Args:
            out_dir (str): The format folder
            results (list): The ACM test results
        Returns:
            None
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.format(len(results), len(self)))
        for process in self.save:
            process(out_dir, results, self.data_infos)

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            eval_results (dict):
        """

        eval_results = dict()
        for process in self.eval:
            sub_eval_results = process(results, self.data_infos)
            eval_results.update(sub_eval_results)

        return eval_results
