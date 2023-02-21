# Copyright (c) Open-MMLab. All rights reserved.
import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size

from .timer import Timer


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, method_num=0, bar_width=50, start=True, file=sys.stdout):
        self.method_num = method_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.method_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.method_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_methods=1):
        assert num_methods > 0
        self.completed += num_methods
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.method_num > 0:
            percentage = self.completed / float(self.method_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.method_num}, ' \
                  f'{fps:.1f} method/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} methods/s')
        self.file.flush()


def track_progress(func, methods, bar_width=50, file=sys.stdout, **kwargs):
    """Track the progress of methods execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each method.
        methods (list or tuple[Iterable, int]): A list of methods or
            (methods, total num).
        bar_width (int): Width of progress bar.

    Returns:
        list: The method results.
    """
    if isinstance(methods, tuple):
        assert len(methods) == 2
        assert isinstance(methods[0], Iterable)
        assert isinstance(methods[1], int)
        method_num = methods[1]
        methods = methods[0]
    elif isinstance(methods, Iterable):
        method_num = len(methods)
    else:
        raise TypeError(
            '"methods" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(method_num, bar_width, file=file)
    results = []
    for method in methods:
        results.append(func(method, **kwargs))
        prog_bar.update()
    prog_bar.file.write('\n')
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(func,
                            methods,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True,
                            file=sys.stdout):
    """Track the progress of parallel method execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    methods are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each method.
        methods (list or tuple[Iterable, int]): A list of methods or
            (methods, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The method results.
    """
    if isinstance(methods, tuple):
        assert len(methods) == 2
        assert isinstance(methods[0], Iterable)
        assert isinstance(methods[1], int)
        method_num = methods[1]
        methods = methods[0]
    elif isinstance(methods, Iterable):
        method_num = len(methods)
    else:
        raise TypeError(
            '"methods" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    method_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(method_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, methods, chunksize)
    else:
        gen = pool.imap_unordered(func, methods, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results


def track_iter_progress(methods, bar_width=50, file=sys.stdout):
    """Track the progress of methods iteration or enumeration with a progress
    bar.

    Tasks are yielded with a simple for-loop.

    Args:
        methods (list or tuple[Iterable, int]): A list of methods or
            (methods, total num).
        bar_width (int): Width of progress bar.

    Yields:
        list: The method results.
    """
    if isinstance(methods, tuple):
        assert len(methods) == 2
        assert isinstance(methods[0], Iterable)
        assert isinstance(methods[1], int)
        method_num = methods[1]
        methods = methods[0]
    elif isinstance(methods, Iterable):
        method_num = len(methods)
    else:
        raise TypeError(
            '"methods" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(method_num, bar_width, file=file)
    for method in methods:
        yield method
        prog_bar.update()
    prog_bar.file.write('\n')
