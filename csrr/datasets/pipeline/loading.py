import os.path as osp
import pickle
import zlib

import h5py
import numpy as np

from ..builder import PIPELINES
from ..utils import Constellation


def normalize_iq_or_ap(x):
    x = (x - np.mean(x, axis=1).reshape(2, 1)) / np.std(x, axis=1).reshape(2, 1)
    return x


@PIPELINES.register_module()
class LoadIQFromFile:
    def __init__(self,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.is_squeeze = is_squeeze,
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        file_path = osp.join(results['data_root'], results['iq_folder'], results['filename'])
        iq = np.load(file_path)
        if self.to_norm:
            iq = normalize_iq_or_ap(iq)
        if self.to_float32:
            iq = iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)
        results['iqs'] = iq
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAPFromFile:
    def __init__(self,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        file_path = osp.join(results['data_root'], results['ap_folder'], results['filename'])
        ap = np.load(file_path)
        if self.to_norm:
            ap = normalize_iq_or_ap(ap)
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the ap as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadConstellationFromFile:
    def __init__(self,
                 filter_config,
                 to_float32=False):
        self.filter_config = filter_config
        self.to_float32 = to_float32

    def __call__(self, results):
        file_path = osp.join(results['data_root'], results['co_folder'], self.filter_config, results['filename'])
        co = np.load(file_path)
        if self.to_float32:
            co = co.astype(np.float32)

        # make a constellation map (square matrix) as a three-dimensional tensor [1, R, R]
        co = np.expand_dims(co, axis=0)
        results['cos'] = co.reshape(1, )
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'filter_config={self.filter_config},'
                    f'to_float32={self.to_float32}, )')
        return repr_str


@PIPELINES.register_module()
class LoadConstellationFromIQFile:
    def __init__(self,
                 filter_size=None,
                 filter_stride=None,
                 to_float32=False):
        if filter_size is None:
            filter_size = [0.02]
        if filter_stride is None:
            filter_stride = [0.02]
        self.convert_tool = Constellation(filter_size, filter_stride)
        self.to_float32 = to_float32

    def __call__(self, results):
        file_path = osp.join(results['data_root'], results['iq_folder'], results['filename'])
        iq = np.load(file_path)
        co, _ = self.convert_tool.generate_by_filter(iq)
        co = co[0]
        if self.to_float32:
            co = co.astype(np.float32)

        # make the iq as a three-dimensional tensor [1, 2, L]
        co = np.expand_dims(co, axis=0)
        results['cos'] = co
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'filter_size={self.filter_size},'
                    f'filter_stride={self.filter_stride},'
                    f'to_float32={self.to_float32}, )')
        return repr_str


@PIPELINES.register_module()
class LoadIQFromCache:
    def __init__(self,
                 data_root,
                 filename,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.filename = filename
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', filename), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['filename']]
        iq = self.cache_data['data'][idx]
        if self.to_norm:
            iq = normalize_iq_or_ap(iq)
        if self.to_float32:
            iq = iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)
        results['iqs'] = iq
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAPFromCache:
    def __init__(self,
                 data_root,
                 filename,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.filename = filename
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', filename), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['filename']]
        ap = self.cache_data['data'][idx]
        if self.to_norm:
            ap = normalize_iq_or_ap(ap)
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadConstellationFromCache:
    def __init__(self,
                 data_root,
                 filename,
                 to_float32=False):
        self.data_root = data_root
        self.filename = filename
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', filename), 'rb'))
        self.to_float32 = to_float32

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['filename']]
        co = self.cache_data['data'][idx]
        co = zlib.decompress(co)
        co = np.frombuffer(co, dtype=self.cache_data['dtype']).reshape(self.cache_data['shape'])
        if self.to_float32:
            co = co.astype(np.float32)

        # make the iq as a three-dimensional tensor [1, R, R]
        co = np.expand_dims(co, axis=0)
        results['cos'] = co
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, )')
        return repr_str


@PIPELINES.register_module()
class LoadIQFromHDF5:
    def __init__(self,
                 data_root,
                 filename,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.filename = filename
        hf = h5py.File(osp.join(data_root, 'hdf5', filename), 'r')
        self.hdf5_data = hf['iq']
        self.lookup_table = pickle.load(
            open(osp.join(data_root, 'hdf5', osp.basename(filename).split('.')[0] + '.pkl'), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.lookup_table[results['filename']]
        iq = self.hdf5_data[idx, :, :]
        if self.to_norm:
            iq = normalize_iq_or_ap(iq)
        if self.to_float32:
            iq = iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)
        results['iqs'] = iq
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAPFromHDF5:
    def __init__(self,
                 data_root,
                 filename,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.filename = filename
        hf = h5py.File(osp.join(data_root, 'hdf5', filename), 'r')
        self.hdf5_data = hf['ap']
        self.lookup_table = pickle.load(
            open(osp.join(data_root, 'hdf5', osp.basename(filename).split('.')[0] + '.pkl'), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.lookup_table[results['filename']]
        ap = self.hdf5_data[idx, :, :]
        if self.to_norm:
            ap = normalize_iq_or_ap(ap)
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadConstellationFromIQCache:
    def __init__(self,
                 data_root,
                 filename,
                 filter_size=None,
                 filter_stride=None,
                 to_float32=False):
        if filter_size is None:
            filter_size = [0.02]
        if filter_stride is None:
            filter_stride = [0.02]
        self.data_root = data_root
        self.filename = filename
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', filename), 'rb'))
        self.convert_tool = Constellation(filter_size, filter_stride)
        self.to_float32 = to_float32

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['filename']]
        iq = self.cache_data['data'][idx]
        co = self.convert_tool.generate_by_filter(iq)
        if self.to_float32:
            co = co.astype(np.float32)

        # make the iq as a three-dimensional tensor [1, 2, L]
        co = np.expand_dims(co, axis=0)
        results['cos'] = co
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'filter_size={self.filter_size},'
                    f'filter_stride={self.filter_stride},'
                    f'to_float32={self.to_float32}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAPFromIQ:
    def __init__(self,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        iq = results['iqs']
        iq = iq[0, :, :]
        amplitude = np.sqrt(np.sum(np.power(iq, 2), axis=0))
        phase = np.arctan(iq[0, :] / (iq[1, :] + np.finfo(np.float32).eps))
        ap = np.vstack((amplitude, phase))
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadFTFromIQ:
    def __init__(self,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        iq = results['iqs']
        iq = iq[0, :, :]
        iq = iq[0, :] + 1j * iq[1, :]
        ft = np.fft.fft(iq)
        amplitude = np.abs(ft)
        phase = np.angle(ft)
        ft = np.vstack((amplitude, phase))
        if self.to_float32:
            ft = ft.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            ft = np.expand_dims(ft, axis=0)
        results['fts'] = ft
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'filename={self.filename},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    def __init__(self, with_mod=True, with_snr=False, with_dev=False):
        self.with_mode = with_mod
        self.with_snr = with_snr
        self.with_dev = with_dev

    def __call__(self, results):
        if self.with_mode:
            results['mod_labels'] = np.array(results['item_mod_label'], dtype=np.int64)
        if self.with_snr:
            results['snr_labels'] = np.array(results['item_snr_label'], dtype=np.int64)
        if self.with_dev:
            results['dev_labels'] = np.array(results['item_dev_label'], dtype=np.int64)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_mod={self.with_mode}, '
        repr_str += f'with_snr={self.with_snr})'

        return repr_str
