#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Project: ChangShuoRadioRecognition
File: __init__.py.py
Author: Citybuster
Time: 2021/8/23 10:45
Email: chagshuo@bupt.edu.cn
"""
from .configs import _COLORS, _MARKERS, _LINES
from .legend import generate_legend_config
from .scatter import generate_scatter_config

__all__ = [
    'generate_legend_config',
    'generate_scatter_config',
    '_COLORS', '_MARKERS', '_LINES'
]
