import os.path as osp
import pickle
import zlib
from typing import Dict

import h5py
import numpy as np

from ..builder import PIPELINES


def normalize_iq_or_ap(x):
    x = (x - np.mean(x, axis=1).reshape(2, 1)) / np.std(x, axis=1).reshape(2, 1)
    return x


class Constellation:
    def __init__(self, filter_size=None, filter_stride=None):
        # matrix window info
        self.height_range = [-1, 1]
        self.width_range = [-1, 1]

        # parameters for converting sequence
        # data (2, N) to constellation matrix based on conv mode
        if filter_size is None:
            self.filter_size = [0.05, 0.02]
        else:
            self.filter_size = filter_size
        if filter_stride is None:
            self.filter_stride = [0.05, 0.02]
        else:
            self.filter_stride = filter_stride

    def get_filters(self):
        filters = []
        for filter_size, filter_stride in zip(self.filter_size, self.filter_stride):
            filters.append([filter_size, filter_stride])

        return filters

    def generate_by_filter(self, data):

        constellations = []
        filters = []
        for filter_size, filter_stride in zip(self.filter_size, self.filter_stride):
            matrix_width = int((self.width_range[1] - self.width_range[0] - filter_size) / filter_stride + 1)
            matrix_height = int((self.height_range[1] - self.height_range[0] - filter_size) / filter_stride + 1)

            constellation = np.zeros((matrix_height, matrix_width))

            def axis_is(query_axis_x, query_axis_y):
                axis_x = query_axis_x // filter_stride
                axis_y = query_axis_y // filter_stride
                if axis_x * filter_stride + filter_size < query_axis_x:
                    position = [None, None]
                elif axis_y * filter_stride + filter_size < query_axis_y:
                    position = [None, None]
                else:
                    position = [int(axis_x), int(axis_y)]
                return position

            pos_list = map(axis_is, list(data[0, :]), list(data[1, :]))
            num_point = 0
            for pos in pos_list:
                if pos[0] is not None:
                    constellation[pos[0], pos[1]] += 1
                    num_point += 1
            constellations.append(constellation / num_point)
            filters.append([filter_size, filter_stride])

        return constellations, filters


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
        file_path = osp.join(results['data_root'], results['iq_folder'], results['file_name'])
        iq = np.load(file_path)
        if self.to_norm:
            iq = normalize_iq_or_ap(iq)
        if self.to_float32:
            iq = iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)
        results['inputs']['iqs'] = iq
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
        file_path = osp.join(results['data_root'], results['ap_folder'], results['file_name'])
        ap = np.load(file_path)
        if self.to_norm:
            ap = normalize_iq_or_ap(ap)
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the ap as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['inputs']['aps'] = ap
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
        file_path = osp.join(results['data_root'], results['co_folder'], self.filter_config, results['file_name'])
        co = np.load(file_path)
        if self.to_float32:
            co = co.astype(np.float32)

        # make a constellation map (square matrix) as a three-dimensional tensor [1, R, R]
        co = np.expand_dims(co, axis=0)
        results['inputs']['cos'] = co.reshape(1, )
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
        file_path = osp.join(results['data_root'], results['iq_folder'], results['file_name'])
        iq = np.load(file_path)
        co, _ = self.convert_tool.generate_by_filter(iq)
        co = co[0]
        if self.to_float32:
            co = co.astype(np.float32)

        # make the iq as a three-dimensional tensor [1, 2, L]
        co = np.expand_dims(co, axis=0)
        results['inputs']['cos'] = co
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
                 file_name,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.file_name = file_name
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', file_name), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['file_name']]
        iq = self.cache_data['data'][idx]
        if self.to_norm:
            iq = normalize_iq_or_ap(iq)
        if self.to_float32:
            iq = iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)
        results['inputs']['iqs'] = iq
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAPFromCache:
    def __init__(self,
                 data_root,
                 file_name,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.file_name = file_name
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', file_name), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['file_name']]
        ap = self.cache_data['data'][idx]
        if self.to_norm:
            ap = normalize_iq_or_ap(ap)
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['inputs']['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadConstellationFromCache:
    def __init__(self,
                 data_root,
                 file_name,
                 to_float32=False):
        self.data_root = data_root
        self.file_name = file_name
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', file_name), 'rb'))
        self.to_float32 = to_float32

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['file_name']]
        co = self.cache_data['data'][idx]
        co = zlib.decompress(co)
        co = np.frombuffer(co, dtype=self.cache_data['dtype']).reshape(self.cache_data['shape'])
        if self.to_float32:
            co = co.astype(np.float32)

        # make the iq as a three-dimensional tensor [1, R, R]
        co = np.expand_dims(co, axis=0)
        results['inputs']['cos'] = co
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
                    f'to_float32={self.to_float32}, )')
        return repr_str


@PIPELINES.register_module()
class LoadIQFromHDF5:
    def __init__(self,
                 data_root,
                 file_name,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.file_name = file_name
        hf = h5py.File(osp.join(data_root, 'hdf5', file_name), 'r')
        self.hdf5_data = hf['iq']
        self.lookup_table = pickle.load(
            open(osp.join(data_root, 'hdf5', osp.basename(file_name).split('.')[0] + '.pkl'), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.lookup_table[results['file_name']]
        iq = self.hdf5_data[idx, :, :]
        if self.to_norm:
            iq = normalize_iq_or_ap(iq)
        if self.to_float32:
            iq = iq.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            iq = np.expand_dims(iq, axis=0)
        results['inputs']['iqs'] = iq
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAPFromHDF5:
    def __init__(self,
                 data_root,
                 file_name,
                 is_squeeze=False,
                 to_float32=False,
                 to_norm=False):
        self.data_root = data_root
        self.file_name = file_name
        hf = h5py.File(osp.join(data_root, 'hdf5', file_name), 'r')
        self.hdf5_data = hf['ap']
        self.lookup_table = pickle.load(
            open(osp.join(data_root, 'hdf5', osp.basename(file_name).split('.')[0] + '.pkl'), 'rb'))
        self.is_squeeze = is_squeeze
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        idx = self.lookup_table[results['file_name']]
        ap = self.hdf5_data[idx, :, :]
        if self.to_norm:
            ap = normalize_iq_or_ap(ap)
        if self.to_float32:
            ap = ap.astype(np.float32)

        if not self.is_squeeze:
            # make the iq as a three-dimensional tensor [1, 2, L]
            ap = np.expand_dims(ap, axis=0)
        results['inputs']['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadConstellationFromIQCache:
    def __init__(self,
                 data_root,
                 file_name,
                 filter_size=None,
                 filter_stride=None,
                 to_float32=False):
        if filter_size is None:
            filter_size = [0.02]
        if filter_stride is None:
            filter_stride = [0.02]
        self.data_root = data_root
        self.file_name = file_name
        self.cache_data = pickle.load(open(osp.join(data_root, 'cache', file_name), 'rb'))
        self.convert_tool = Constellation(filter_size, filter_stride)
        self.to_float32 = to_float32

    def __call__(self, results):
        idx = self.cache_data['lookup_table'][results['file_name']]
        iq = self.cache_data['data'][idx]
        co = self.convert_tool.generate_by_filter(iq)
        if self.to_float32:
            co = co.astype(np.float32)

        # make the iq as a three-dimensional tensor [1, 2, L]
        co = np.expand_dims(co, axis=0)
        results['inputs']['cos'] = co
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
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
        self._cache = dict()

    def __call__(self, results):
        if results['file_name'] in self._cache:
            results['inputs']['aps'] = self._cache[results['file_name']]
        else:
            iq = results['inputs']['iqs']
            iq = iq[0, :, :]
            amplitude = np.sqrt(np.sum(np.power(iq, 2), axis=0))
            phase = np.arctan(iq[0, :] / (iq[1, :] + np.finfo(np.float32).eps))
            ap = np.vstack((amplitude, phase))
            if self.to_float32:
                ap = ap.astype(np.float32)

            if not self.is_squeeze:
                # make the iq as a three-dimensional tensor [1, 2, L]
                ap = np.expand_dims(ap, axis=0)
            self._cache[results['file_name']] = ap
            results['inputs']['aps'] = ap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'is_squeeze={self.is_squeeze},'
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
        results['inputs']['fts'] = ft
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'data_root={self.data_root},'
                    f'file_name={self.file_name},'
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    def __init__(self, target_info: Dict[str, np.dtype]):
        self.target_info = target_info
        self.numpy_type_look_table = {'int64': np.int64, 'float32': np.float32}

    def __call__(self, results):
        for target_name in self.target_info:
            target_type = self.numpy_type_look_table[self.target_info[target_name]]
            if isinstance(results[target_name], np.ndarray):
                results['targets'][f'{target_name}s'] = results[target_name].astype(target_type)
            elif isinstance(results[target_name], list):
                results['targets'][f'{target_name}s'] = np.array(results[target_name], target_type)
            elif isinstance(results[target_name], tuple):
                results['targets'][f'{target_name}s'] = np.array(results[target_name], target_type)
            elif type(results[target_name]) == int or type(results[target_name]) == float:
                results['targets'][f'{target_name}s'] = np.array(results[target_name], target_type)
            else:
                raise TypeError(
                    f'The target data type {type(results[target_name])} of {target_name} is not supported, '
                    f'the supported types are np.ndarray, list or tuple with int or float vars, int or float')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(targets={self.target_info})'

        return repr_str


@PIPELINES.register_module()
class MLDNNSNRLabel:
    def __init__(self, snr_threshold=0):
        self.snr_threshold = snr_threshold

    def __call__(self, results):
        snr = results['snr']
        if snr >= self.snr_threshold:
            results['targets']['snrs'] = np.array(0, np.int64)
        else:
            results['targets']['snrs'] = np.array(1, np.int64)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(snr_threshold={self.snr_threshold})'

        return repr_str
