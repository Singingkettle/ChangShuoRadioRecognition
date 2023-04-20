from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DeepSigDataset(CustomDataset):
    """Custom dataset for modulation classification.
    Args:
        ann_file (str): Annotation file path.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, ``Amplitude/Phase data``,
                                   ``constellation data``.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    CLASSES = None
    SNRS = None

    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False,
                 preprocess=None, evaluate=None, format=None):
        super(DeepSigDataset, self).__init__(ann_file, pipeline, data_root, test_mode, preprocess, evaluate, format)

        self.CLASSES = self.data_infos['modulations']
        self.SNRS = self.data_infos['snrs']

    def get_ann_info(self, idx):
        results = dict()
        results['snr'] = self.data_infos['annotations'][idx]['snr']
        results['file_name'] = self.data_infos['annotations'][idx]['file_name']
        results['modulation'] = self.CLASSES.index(self.data_infos['annotations'][idx]['modulation'])
        results['snrs'] = self.SNRS
        results['modulations'] = self.CLASSES

        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['iq_folder'] = 'sequence_data/iq'
        results['ap_folder'] = 'sequence_data/ap'
        results['co_folder'] = 'constellation_data'
        results['inputs'] = dict()
        return results
