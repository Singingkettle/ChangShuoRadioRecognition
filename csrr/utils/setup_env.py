# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in csrr into the registries.

    Args:
        init_default_scope (bool): Whether initialize the csrr default
            scope. If True, the global default scope will be set to
            `csrr`, and all registries will build modules from
            csrr's registry node. To understand more about the registry,
            please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa: E501
    import csrr.datasets  # noqa: F401,F403
    import csrr.engine  # noqa: F401,F403
    import csrr.evaluation  # noqa: F401,F403
    import csrr.models  # noqa: F401,F403
    import csrr.structures  # noqa: F401,F403
    import csrr.visualization  # noqa: F401,F403

    if not init_default_scope:
        return

    current_scope = DefaultScope.get_current_instance()
    if current_scope is None:
        DefaultScope.get_instance('csrr', scope_name='csrr')
    elif current_scope.scope_name != 'csrr':
        warnings.warn(
            f'The current default scope "{current_scope.scope_name}" '
            'is not "csrr", `register_all_modules` will force '
            'the current default scope to be "csrr". If this is '
            'not expected, please set `init_default_scope=False`.')
        # avoid name conflict
        new_instance_name = f'csrr-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='csrr')
