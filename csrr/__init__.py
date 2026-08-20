import mmengine
from mmengine.utils import digit_version

from .apis import *  # noqa: F401, F403
from .version import __version__

mmengine_minimum_version = '0.8.3'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (digit_version(mmengine_minimum_version) <= mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

__all__ = ['__version__']
