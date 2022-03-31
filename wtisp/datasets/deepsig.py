from .builder import DATASETS, build_augment, build_evaluate, build_save
from .custom import CustomAMCDataset
from .utils import format_results
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class DeepSigDataset(CustomAMCDataset):
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
                 augment=None, evaluate=None, save=None):
        super(DeepSigDataset, self).__init__(ann_file, pipeline, data_root, test_mode)
        if augment is None:
            self.augment = None
        else:
            self.augment = build_augment(augment)
            for process in self.augment:
                self.data_infos = process(self.data_infos)
        if evaluate is None:
            self.eval = None
        else:
            self.eval = build_evaluate(evaluate)
        if save is None:
            self.save = None
        else:
            self.save = build_save(save)
        self.CLASSES, self.SNRS = self.extract_CLASSES_SNRS(self.data_infos)

    def load_annotations(self, ann_file):
        data_infos = IOLoad(ann_file)
        # there is a bug when using json package to save the dict var, where its key is int
        # after saving, the int key will be transferred as string
        # please refer to:
        # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
        label_to_mod = {int(key): value for key, value in data_infos['label_to_mod'].items()}
        label_to_snr = {int(key): value for key, value in data_infos['label_to_snr'].items()}
        data_infos['label_to_mod'] = label_to_mod
        data_infos['label_to_snr'] = label_to_snr

        return data_infos

    def __len__(self):
        return len(self.data_infos['item_filename'])

    def extract_CLASSES_SNRS(self, data_infos):

        label_to_mod = data_infos['label_to_mod']
        label_to_snr = data_infos['label_to_snr']
        CLASSES = [''] * len(label_to_mod)
        SNRS = [0] * len(label_to_snr)

        for label in label_to_mod:
            CLASSES[label] = label_to_mod[label]
        for label in label_to_snr:
            SNRS[label] = label_to_snr[label]

        return CLASSES, SNRS

    def get_ann_info(self, idx):
        results = dict()
        results['item_mod_label'] = self.data_infos['item_mod_label'][idx]
        results['item_snr_label'] = self.data_infos['item_snr_label'][idx]
        results['item_snr_value'] = self.data_infos['item_snr_value'][idx]
        results['item_snr_index'] = self.data_infos['item_snr_index'][idx]
        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['iq_folder'] = 'sequence_data/iq'
        results['ap_folder'] = 'sequence_data/ap'
        results['co_folder'] = 'constellation_data'

        return results

    def prepare_train_data(self, idx):
        filename = self.data_infos['item_filename'][idx]
        ann_info = self.get_ann_info(idx)
        results = dict(filename=filename)
        results.update(ann_info)
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        filename = self.data_infos['item_filename'][idx]
        results = dict(filename=filename)
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def format_out(self, out_dir, results):
        """Format results and save."""

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.format(len(results), len(self)))
        results = format_results(results)
        for process in self.save:
            process(out_dir, results, self.data_infos, self.CLASSES, self.SNRS)

    def evaluate(self, results, logger=None):
        results = format_results(results)
        eval_results = dict()
        for process in self.eval:
            sub_eval_results = process(results, self.data_infos)
            eval_results.update(sub_eval_results)

        return eval_results
