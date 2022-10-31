from .builder import DATASETS, build_preprocess, build_evaluate, build_save
from .custom import CustomAMCDataset
from .utils import format_results
from ..common.fileio import load as IOLoad


@DATASETS.register_module()
class WTISPDataset(CustomAMCDataset):
    """Custom dataset for modulation classification.
    Args:
        ann_file (str): Annotation file path.
        data_root (str, optional): Data root for ``ann_file``, ``In-phase/Quadrature data``, ``Amplitude/Phase data``,
                                   ``constellation data``.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """
    CLASSES_MOD = ['BPSK', 'QPSK', 'PSK8', 'QAM16', 'QAM32']
    CLASSES_SEI = ['Device1', 'Device2', 'Device3', 'Device4', 'Device5']
    SNRS = [0, 5, 10, 15, 20, 25, 30]

    def __init__(self, ann_file, pipeline, data_root=None, test_mode=False,
                 preprocess=None, evaluate=None, save=None):
        super(WTISPDataset, self).__init__(ann_file, pipeline, data_root, test_mode)
        if preprocess is None:
            self.preprocess = None
        else:
            self.preprocess = build_preprocess(preprocess)
            for process in self.preprocess:
                self.data_infos = process(self.data_infos)
                # set group flag for the sampler
                if not self.test_mode:
                    self._set_group_flag()
        if evaluate is None:
            self.eval = None
        else:
            self.eval = build_evaluate(evaluate)
        if save is None:
            self.save = None
        else:
            self.save = build_save(save)

    def load_annotations(self, ann_file):
        data_infos = IOLoad(ann_file)
        # there is a bug when using json package to save the dict var, where its key is int
        # after saving, the int key will be transferred as string
        # please refer to:
        # https://stackoverflow.com/questions/1450957/pythons-json-module-converts-int-dictionary-keys-to-strings
        item_snr_value = [int(value) for value in data_infos['item_snr_label_str']]
        snr_to_label = {key: value for value, key in enumerate(self.SNRS)}
        label_to_snr = {value: key for key, value in snr_to_label.items()}
        item_snr_label = [snr_to_label[snr] for snr in item_snr_value]
        label_to_mod = {int(key): value for key, value in data_infos['label_to_mod'].items()}
        label_to_dev = {int(key): value for key, value in data_infos['label_to_dev'].items()}

        data_infos['item_snr_value'] = item_snr_value
        data_infos['item_snr_label'] = item_snr_label
        data_infos['snr_to_label'] = snr_to_label
        data_infos['label_to_snr'] = label_to_snr
        data_infos['snr_to_index'] = snr_to_label
        data_infos['item_snr_index'] = item_snr_label

        data_infos['label_to_dev'] = label_to_dev
        data_infos['label_to_mod'] = label_to_mod
        return data_infos

    def __len__(self):
        return len(self.data_infos['item_filename'])

    def get_ann_info(self, idx):
        results = dict()
        results['item_mod_label'] = self.data_infos['item_mod_label'][idx]
        results['item_dev_label'] = self.data_infos['item_dev_label'][idx]
        results['item_snr_value'] = self.data_infos['item_snr_value'][idx]
        results['item_snr_index'] = self.data_infos['item_snr_index'][idx]
        return results

    def pre_pipeline(self, results):
        results['data_root'] = self.data_root
        results['iq_folder'] = 'iqs'

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
