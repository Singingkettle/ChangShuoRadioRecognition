"""FLOPs / parameter count / inference-time table.

For every (dataset, method) entry in the publish mapping we build the model
from its training config, run ``mmengine.analysis.get_model_complexity_info``
with a synthetic input that matches the shapes specified by the config's
``train_pipeline`` Reshape transforms, and optionally measure inference time.

The output is a markdown table ``flops.md`` saved next to the other figures.

Notes
-----
* ``get_model_complexity_info`` lives in mmengine >= 0.7 and accepts an
  ``input_shape`` (single tensor) **or** ``inputs`` (already-built tensor or
  tuple). We need the latter because several models in this repo (MLDNN,
  FastMLDNN, DSCLDNN, ...) consume a dict of tensors.
* Configs can opt out of the table entry by being absent from ``publish``; this
  class is robust to those cases.
* The timing loop is opt-in via ``measure_time=True`` because it is slow and
  not required for most reports.
"""

import os
import time
from typing import Any, Dict, List, Optional

import torch

from ..builder import TABLES


def _infer_frame_length(cfg) -> int:
    """Best-effort frame length inference from ``cfg.data_root``."""
    data_root = ''
    if hasattr(cfg, 'get'):
        data_root = cfg.get('data_root', '') or ''
    if not data_root:
        for loader_key in ('train_dataloader', 'val_dataloader',
                           'test_dataloader'):
            loader = cfg.get(loader_key) if hasattr(cfg, 'get') else None
            if not loader:
                continue
            ds = loader.get('dataset', {}) if isinstance(loader, dict) else {}
            data_root = (ds.get('data_root', '') if isinstance(ds, dict)
                         else '') or data_root
            if data_root:
                break
    data_root = str(data_root).lower()
    if '2018' in data_root or 'hisar' in data_root:
        return 1024
    return 128


def _apply_pipeline_to_shapes(pipeline, base_shapes: Dict[str, List[int]]):
    """Apply ``Reshape`` and ``Transpose`` steps to ``base_shapes`` in place.

    Other transforms are ignored. Recurses into ``task_handlers`` /
    ``transforms`` / ``pipeline`` nesting.
    """
    if not pipeline:
        return base_shapes
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        step_type = step.get('type')
        if step_type == 'Reshape':
            for key, shape in (step.get('shapes', {}) or {}).items():
                base_shapes[key] = list(shape)
        elif step_type == 'Transpose':
            for key, order in (step.get('orders', {}) or {}).items():
                if key in base_shapes:
                    current = base_shapes[key]
                    base_shapes[key] = [current[i] for i in order]
        elif step_type == 'IQToAP':
            # IQ -> AP transformation keeps the [2, L] layout but renames key.
            if 'iq' in base_shapes:
                base_shapes.setdefault('ap', list(base_shapes['iq']))
        # Recurse into nested keys
        for nested_key in ('task_handlers', 'transforms', 'pipeline'):
            nested = step.get(nested_key)
            if isinstance(nested, dict):
                _apply_pipeline_to_shapes(list(nested.values()), base_shapes)
            elif isinstance(nested, list):
                _apply_pipeline_to_shapes(nested, base_shapes)
    return base_shapes


def _pack_input_keys(pipeline) -> List[str]:
    """Return the ordered list of input keys consumed by the final pack step."""
    if not pipeline:
        return []
    for step in reversed(pipeline):
        if not isinstance(step, dict):
            continue
        if step.get('type') in ('PackInputs', 'PackMultiTaskInputs'):
            keys = step.get('input_key')
            if isinstance(keys, list):
                return list(keys)
            if isinstance(keys, str):
                return [keys]
    return []


def _build_dummy_inputs(cfg, device: torch.device):
    """Construct a dummy input batch matching the model's expected layout.

    Returns a tuple ``(inputs, input_shape_repr)`` where ``inputs`` is what we
    pass to ``model(inputs, mode='tensor')`` and ``input_shape_repr`` is a
    short string used for the markdown table.
    """
    # Prefer test pipeline (closer to inference); fall back to train pipeline.
    pipeline_candidates = []
    for key in ('test_pipeline', 'pipeline', 'train_pipeline'):
        if hasattr(cfg, 'get'):
            pipeline_candidates.append(cfg.get(key))
    for loader_key in ('test_dataloader', 'val_dataloader', 'train_dataloader'):
        loader = cfg.get(loader_key) if hasattr(cfg, 'get') else None
        if loader is None:
            continue
        ds = loader.get('dataset', {}) if isinstance(loader, dict) else {}
        if isinstance(ds, dict):
            pipeline_candidates.append(ds.get('pipeline'))

    frame_length = _infer_frame_length(cfg)
    # Base raw-IQ shape coming out of the dataset before any transform.
    base_shapes: Dict[str, List[int]] = {
        'iq': [2, frame_length],
        'ap': [2, frame_length],
    }

    pack_keys: List[str] = []
    # Only the first non-empty pipeline candidate is used (applying the same
    # Reshape/Transpose multiple times would cancel itself out).
    for p in pipeline_candidates:
        if not p:
            continue
        _apply_pipeline_to_shapes(list(p), base_shapes)
        pack_keys = _pack_input_keys(list(p))
        break

    if not pack_keys:
        # Pick the most likely default
        pack_keys = ['iq'] if 'iq' in base_shapes else list(base_shapes.keys())

    tensors = {
        k: torch.randn((1, *base_shapes[k]), dtype=torch.float32, device=device)
        for k in pack_keys if k in base_shapes
    }

    if not tensors:
        raise RuntimeError(
            'Could not determine input shapes from config '
            '(no Reshape / Transpose found and pack key not in base_shapes).')

    if len(tensors) == 1:
        only_key, only_tensor = next(iter(tensors.items()))
        repr_str = f'{only_key}={tuple(only_tensor.shape)}'
        return only_tensor, repr_str

    repr_str = ', '.join(
        f'{k}={tuple(v.shape)}' for k, v in tensors.items())
    return tensors, repr_str


def _format_count(n: Optional[float], unit_divisor: float, suffix: str) -> str:
    if n is None:
        return '-'
    return '{:.3f} {}'.format(n / unit_divisor, suffix)


@TABLES.register_module()
class Flops:
    """Generate a complexity table for every published method.

    Args:
        dataset (Dict[str, List[str]]): Maps ``dataset_name -> [method_name,
            ...]``. Methods must be present in ``info['publish'][dataset_name]``
            so the table can reach the corresponding training config.
        measure_time (bool): If True, measure inference latency. Default False.
        timing_iters (int): Number of forward passes used for timing.
        warmup_iters (int): Number of warm-up forward passes.
        device (str): ``'cpu'`` or ``'cuda'``. Default ``'cpu'`` to avoid
            requiring a GPU when generating reports.
        legend (Any): Accepted for builder symmetry; unused.
        scatter (Any): Accepted for builder symmetry; unused.
    """

    def __init__(self,
                 dataset: Dict[str, List[str]],
                 measure_time: bool = False,
                 timing_iters: int = 30,
                 warmup_iters: int = 5,
                 device: str = 'cpu',
                 legend: Any = None,
                 scatter: Any = None):
        self.dataset = dataset
        self.measure_time = measure_time
        self.timing_iters = max(1, int(timing_iters))
        self.warmup_iters = max(0, int(warmup_iters))
        self.device = device

    def __call__(self, performances, save_dir):  # noqa: ARG002
        os.makedirs(save_dir, exist_ok=True)
        content = '# Complexity Table  \n'

        device = torch.device(self.device) if torch.cuda.is_available() or \
            self.device == 'cpu' else torch.device('cpu')

        publish = self._resolve_publish(performances)
        work_dir = self._resolve_work_dir(performances)

        for dataset_name, method_names in self.dataset.items():
            content += f'## Dataset {dataset_name}  \n'
            header = '| Method | Params (M) | FLOPs (M) | Input '
            sep = '|:---:|:---:|:---:|:---:'
            if self.measure_time:
                header += '| Inference time (ms/sample) '
                sep += '|:---:'
            header += '|  \n'
            sep += '|  \n'
            content += header + sep

            for method_name in method_names:
                row = self._row_for(method_name, dataset_name, publish,
                                    work_dir, device)
                content += row

        save_path = os.path.join(save_dir, 'flops.md')
        with open(save_path, 'w') as f:
            f.write(content)
        print(f'Save: {save_path}')

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _resolve_publish(performances):
        if isinstance(performances, dict) and '_info' in performances:
            return performances['_info'].get('publish', {})
        return {}

    @staticmethod
    def _resolve_work_dir(performances):
        if isinstance(performances, dict) and '_info' in performances:
            return performances['_info'].get('work_dir', 'work_dirs')
        return 'work_dirs'

    def _row_for(self, method_name, dataset_name, publish, work_dir, device):
        cfg_subdir = publish.get(dataset_name, {}).get(method_name)
        if cfg_subdir is None:
            return f'| {method_name} | - | - | - {" | -" if self.measure_time else ""} |  \n'

        cfg_path = os.path.join(work_dir, cfg_subdir, f'{cfg_subdir}.py')
        if not os.path.isfile(cfg_path):
            # Nested layout (work_dir/<model>/<dataset>/): the training run
            # dumps the resolved config into the run dir under its original
            # basename, so glob for it rather than assuming subdir == basename.
            import glob
            dumped = sorted(glob.glob(os.path.join(work_dir, cfg_subdir, '*.py')))
            cfg_path = dumped[0] if dumped else self._search_config(cfg_subdir)

        if cfg_path is None or not os.path.isfile(cfg_path):
            print(f'[Flops] Skipping {method_name}: config not found '
                  f'(looked under work_dir/{cfg_subdir}/{cfg_subdir}.py and '
                  f'configs/**/{cfg_subdir}.py).')
            extra = ' | -' if self.measure_time else ''
            return f'| {method_name} | - | - | not found{extra} |  \n'

        try:
            from mmengine.config import Config
            cfg = Config.fromfile(cfg_path)
        except Exception as exc:  # noqa: BLE001
            print(f'[Flops] Failed to load config for {method_name}: {exc}')
            extra = ' | -' if self.measure_time else ''
            return f'| {method_name} | - | - | load error{extra} |  \n'

        try:
            from csrr.registry import MODELS
            from mmengine.registry import init_default_scope
            init_default_scope(cfg.get('default_scope', 'csrr'))
            model = MODELS.build(cfg.model)
        except Exception as exc:  # noqa: BLE001
            print(f'[Flops] Failed to build model for {method_name}: {exc}')
            extra = ' | -' if self.measure_time else ''
            return f'| {method_name} | - | - | build error{extra} |  \n'

        model = model.to(device).eval()

        try:
            inputs, input_repr = _build_dummy_inputs(cfg, device)
        except Exception as exc:  # noqa: BLE001
            print(f'[Flops] Failed to construct dummy input for '
                  f'{method_name}: {exc}')
            extra = ' | -' if self.measure_time else ''
            return f'| {method_name} | - | - | shape error{extra} |  \n'

        params, flops = self._compute_complexity(model, inputs, method_name)
        params_str = _format_count(params, 1e6, 'M')
        flops_str = _format_count(flops, 1e6, 'M')

        row = f'| {method_name} | {params_str} | {flops_str} | {input_repr} '
        if self.measure_time:
            ms = self._measure_time(model, inputs)
            row += '| {:.3f} '.format(ms) if ms is not None else '| - '
        row += '|  \n'

        # Free memory before next iteration
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return row

    @staticmethod
    def _search_config(cfg_subdir):
        """Look for ``configs/**/<cfg_subdir>.py`` as a fallback."""
        import glob
        candidates = glob.glob(os.path.join('configs', '**', f'{cfg_subdir}.py'),
                               recursive=True)
        return candidates[0] if candidates else None

    def _compute_complexity(self, model, inputs, method_name):
        """Return ``(params, flops)`` using mmengine.analysis if possible."""
        try:
            from mmengine.analysis import get_model_complexity_info
        except Exception as exc:  # noqa: BLE001
            print(f'[Flops] mmengine.analysis unavailable ({exc}); using '
                  f'parameter count only for {method_name}.')
            params = sum(p.numel() for p in model.parameters())
            return params, None

        # mmengine.analysis expects the model to behave like a vanilla nn.Module
        # in forward mode. SignalClassifier supports ``mode='tensor'`` so we
        # wrap it in a thin shim that forwards to that mode.
        class _Shim(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, *args):
                if len(args) == 1:
                    return self.m(args[0], mode='tensor')
                # mmengine.analysis can pass multiple positional tensors; map
                # them back to a dict if the underlying model expects one.
                return self.m(args, mode='tensor')

        shim = _Shim(model).eval()

        try:
            if isinstance(inputs, dict):
                info = get_model_complexity_info(shim, inputs=(inputs,))
            elif isinstance(inputs, torch.Tensor):
                info = get_model_complexity_info(shim, inputs=(inputs,))
            else:
                info = get_model_complexity_info(shim, inputs=tuple(inputs))
        except Exception as exc:  # noqa: BLE001
            print(f'[Flops] get_model_complexity_info failed for '
                  f'{method_name}: {exc}; falling back to parameter count.')
            params = sum(p.numel() for p in model.parameters())
            return params, None

        # ``info`` is a dict with both numeric and human-readable fields.
        flops = info.get('flops') if isinstance(info, dict) else None
        params = info.get('params') if isinstance(info, dict) else None
        if params is None:
            params = sum(p.numel() for p in model.parameters())
        return params, flops

    def _measure_time(self, model, inputs):
        try:
            with torch.no_grad():
                for _ in range(self.warmup_iters):
                    _ = model(inputs, mode='tensor')
                t0 = time.perf_counter()
                for _ in range(self.timing_iters):
                    _ = model(inputs, mode='tensor')
                t1 = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            print(f'[Flops] Timing forward failed: {exc}')
            return None
        return (t1 - t0) * 1000.0 / self.timing_iters
