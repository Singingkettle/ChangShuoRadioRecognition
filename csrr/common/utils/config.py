# Copyright (c) Open-MMLab. All rights reserved.
import ast
import glob
import json
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
from argparse import Action, ArgumentParser
from collections import abc
from collections import defaultdict
from importlib import import_module

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from .path import glob

if platform.system() == 'Windows':
    import regex as re
else:
    import re

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['file_name', 'text', 'pretty_text']


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


class Config:
    """A facility for figure_configs and figure_configs files.

    It supports common file formats as figure_configs: python/json/yaml. The interface
    is the same as a dict object and also allows access figure_configs values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/figure_configs/a.py')
        >>> cfg.file_name
        "/home/kchen/projects/mmcv/tests/data/figure_configs/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/figure_configs/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _validate_py_syntax(file_name):
        with open(file_name, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in figure_configs '
                              f'file {file_name}: {e}')

    @staticmethod
    def _substitute_predefined_vars(file_name, temp_config_name):
        file_dirname = osp.dirname(file_name)
        file_basename = osp.basename(file_name)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(file_name)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(file_name, 'r') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _file2dict(file_name, use_predefined_variables=True):
        file_name = osp.abspath(osp.expanduser(file_name))
        if not osp.isfile(file_name):
            raise FileNotFoundError('file "{}" does not exist'.format(file_name))

        fileExtname = osp.splitext(file_name)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(file_name,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(file_name, temp_config_file.name)

            if file_name.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(file_name)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
            else:
                raise NotImplementedError('The config file must be a python script.')
            # close temp file
            temp_config_file.close()

        cfg_text = file_name + '\n'
        with open(file_name, 'r') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(file_name)
            base_file_name = cfg_dict.pop(BASE_KEY)
            base_file_name = base_file_name if isinstance(
                base_file_name, list) else [base_file_name]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_file_name:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict `a` into dict `b` (non-inplace). values in `a` will
        # overwrite `b`.
        # copy first to avoid inplace modification
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict):
                    raise TypeError(
                        f'{k}={v} in child figure_configs cannot inherit from base '
                        f'because {k} is a dict in the child figure_configs but is of '
                        f'type {type(b[k])} in base figure_configs. You may set '
                        f'`{DELETE_KEY}=True` to ignore the base figure_configs')
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(file_name, use_predefined_variables=True):
        cfg_dict, cfg_text = Config._file2dict(file_name,
                                               use_predefined_variables)
        return Config(cfg_dict, cfg_text=cfg_text, file_name=file_name)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from figure_configs file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, cfg_text=None, file_name=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for figure_configs file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_file_name', file_name)
        if cfg_text:
            text = cfg_text
        elif file_name:
            with open(file_name, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def file_name(self):
        return self._file_name

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f'Config (path: {self.file_name}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._file_name, self._text)

    def __setstate__(self, state):
        _cfg_dict, _file_name, _text = state
        super(Config, self).__setattr__('_cfg_dict', _cfg_dict)
        super(Config, self).__setattr__('_file_name', _file_name)
        super(Config, self).__setattr__('_text', _text)

    def dump(self, file=None):
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict').to_dict()
        if self.file_name.endswith('.py'):
            if file is None:
                return self.pretty_text
            else:
                with open(file, 'w') as f:
                    f.write(self.pretty_text)
        else:
            import mmcv
            if file is None:
                file_format = self.file_name.split('.')[-1]
                return mmcv.dump(cfg_dict, file_format=file_format)
            else:
                mmcv.dump(cfg_dict, file)

    def merge_from_dict(self, options):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        Args:
            options (dict): dict of figure_configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        super(Config, self).__setattr__(
            '_cfg_dict', Config._merge_a_into_b(option_cfg_dict, cfg_dict))


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def filter_config(cfg, is_regeneration=False, mode='test'):
    configs = []
    test_configs = []
    no_test_configs = dict()
    train_configs = dict()
    for dataset in cfg.publish:
        for method_name in cfg.publish[dataset]:
            config_name = cfg.publish[dataset][method_name]
            config_file_path = osp.join(cfg.work_dir, config_name, f'{config_name}.py')
            if osp.isfile(config_file_path):
                if 'feature-based' in config_name:
                    train_configs[config_name] = -1
                else:
                    best_epoch = get_the_best_checkpoint(cfg.work_dir, config_name)
                    if best_epoch > 0:
                        train_configs[config_name] = best_epoch
                        if not osp.isfile(osp.join(cfg.work_dir, config_name, 'format/res.pkl')):
                            no_test_configs[config_name] = best_epoch
                        else:
                            test_configs.append(config_name)

            configs.append(config_name)

    if is_regeneration:
        no_test_configs = train_configs
        no_train_configs = configs
    else:
        no_train_configs = list(set(configs) - set(train_configs.keys()))

    if mode == 'test':
        return no_test_configs
    elif mode == 'train':
        return no_train_configs
    elif mode == 'summary':
        return test_configs
    else:
        raise ValueError('Unknown mod {} for filtering figure_configs'.format(mode))


def load_json_log(json_log):
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict


def get_total_epoch(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            if 'total_epochs' in line:
                line = line.strip()
                total_epoch = int(line.replace('total_epochs = ', ''))
                return total_epoch

    return 0


def get_the_best_checkpoint(work_dir, config_name):
    json_paths = glob(os.path.join(work_dir, config_name), 'json')

    best_epoch = 0
    if len(json_paths) > 0:
        json_paths = sorted(json_paths)
        max_epoch = 0
        final_metric = 'ACC'
        total_epoch = get_total_epoch(json_paths[-1].replace('.json', ''))
        merge_log = {i + 1: 0 for i in range(total_epoch)}
        for json_path in json_paths:
            log_dict = load_json_log(json_path)
            epochs = list(log_dict.keys())
            if len(epochs) == 0:
                continue
            if max(epochs) > max_epoch:
                max_epoch = max(epochs)
            for epoch in epochs:
                try:
                    if log_dict[epoch]['mode'][-1] == 'val':
                        merge_log[epoch] = log_dict[epoch][final_metric][0]
                except:
                    max_epoch = 0
                    break
        best_epoch = max(merge_log, key=merge_log.get)
        if total_epoch != max_epoch:
            best_epoch = 0

    return best_epoch
