import os.path as osp

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipeline import Compose
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class CustomAMCDataset(Dataset):
    """Custom dataset for modulation classification.
    Args:
        ann_file (str): Annotation file path.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, ``Amplitude/Phase data``,
                                   ``constellation data``.
        test_mode (bool, optional): If set True, annotation will not be loaded.

    """
    CLASSES = None
    SNRS = None

    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False):
        self.ann_file = ann_file
        self.pipeline = pipeline
        self.data_root = data_root
        self.test_mode = test_mode

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

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return IOLoad(ann_file)

    def extract_CLASSES_SNRS(self, data_infos):
        """Extract categories' names of CLASSES and SNRS."""

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

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""

    def prepare_train_data(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

    def prepare_test_data(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

    def format_out(self, out_dir, results):
        """Format results and save."""

    def evaluate(self, results, logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        """
