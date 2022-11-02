import importlib
import pkgutil

import torch

if torch.__version__ != 'parrots':

    def load_ext(name, funcs):
        ext = importlib.import_module('csrr.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext


def check_ops_exist():
    ext_loader = pkgutil.find_loader('csrr._ext')
    return ext_loader is not None
