# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .signal import SignalClassifier
from .timm import TimmClassifier

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'SignalClassifier',
    'HuggingFaceClassifier'
]
