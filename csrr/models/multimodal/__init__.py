# Copyright (c) OpenMMLab. All rights reserved.
from csrr.utils.dependency import WITH_MULTIMODAL

if WITH_MULTIMODAL:
    from .blip import *  # noqa: F401,F403
    from .blip2 import *  # noqa: F401,F403
    from .chinese_clip import *  # noqa: F401, F403
    from .flamingo import *  # noqa: F401, F403
    from .ofa import *  # noqa: F401, F403
else:
    from csrr.registry import MODELS
    from csrr.utils.dependency import register_multimodal_placeholder

    register_multimodal_placeholder([
        'Blip2Caption', 'Blip2Retrieval', 'Blip2VQA', 'BlipCaption',
        'BlipNLVR', 'BlipRetrieval', 'BlipGrounding', 'BlipVQA', 'Flamingo',
        'OFA', 'ChineseCLIP'
    ], MODELS)
