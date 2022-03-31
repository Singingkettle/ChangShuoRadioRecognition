from .builder import DATASETS, build_augment, build_evaluate, build_save
from .custom import CustomAMCDataset
from .utils import format_results
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class OnlineDataset(CustomAMCDataset):
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
        super(OnlineDataset, self).__init__(ann_file, pipeline, data_root, test_mode)
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
        self.CLASSES = self.extract_CLASSES(self.data_infos)

    def load_annotations(self, ann_file):
        data_infos = IOLoad(ann_file)
        data_infos['mod_to_label'] = data_infos['mods']
        label_to_mod = {value: key for key, value in data_infos['mod_to_label'].items()}
        data_infos['label_to_mod'] = label_to_mod

        data_infos['item_filename'] = []
        data_infos['item_mod_label'] = []
        for item in data_infos['data']:
            data_infos['item_filename'].append(item['filename'])
            data_infos['item_mod_label'].append(data_infos['mod_to_label'][item['ann']['labels']])

        data_infos.pop('mods', None)
        data_infos.pop('data', None)

        return data_infos

    def __len__(self):
        return len(self.data_infos['item_filename'])

    def extract_CLASSES(self, data_infos):

        label_to_mod = data_infos['label_to_mod']
        CLASSES = [''] * len(label_to_mod)

        for label in label_to_mod:
            CLASSES[label] = label_to_mod[label]

        return CLASSES

    def get_ann_info(self, idx):
        results = dict()
        results['item_mod_label'] = self.data_infos['item_mod_label'][idx]
        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['iq_folder'] = 'iq'

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
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))
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
