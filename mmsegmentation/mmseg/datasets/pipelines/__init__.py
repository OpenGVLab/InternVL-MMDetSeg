# Copyright (c) OpenMMLab. All rights reserved.
from .compose import Compose
from .formatting import (Collect, ImageToTensor, ToDataContainer, ToTensor,
                         Transpose, to_tensor)
from .loading import LoadAnnotations, LoadImageFromFile,LoadDataFromFile
from .test_time_aug import MultiScaleFlipAug
from .transforms import (CLAHE, AdjustGamma, Normalize, Pad,
                         PhotoMetricDistortion, RandomCrop, RandomCutOut, RandomPadCrop, CenterPad,
                         RandomFlip, RandomMosaic, RandomRotate, Rerange,
                         Resize, RGB2Gray, SegRescale)
from .formatting import ToMask
from .transforms import SETR_Resize, PadShortSide, MapillaryHack

from .custom_transforms import DNANetTransforms

__all__ = [
     # SIRST ----
    'DNANetTransforms',
    'CollectToList',
    "LoadDataFromFile",
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'LoadAnnotations', 'LoadImageFromFile',
    'MultiScaleFlipAug', 'Resize', 'RandomFlip', 'Pad', 'RandomCrop',
    'Normalize', 'SegRescale', 'PhotoMetricDistortion', 'RandomRotate',
    'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray', 'RandomCutOut',
    'RandomMosaic', 'ToMask', 'SETR_Resize', 'PadShortSide', 'MapillaryHack',
    "RandomPadCrop", "CenterPad"
]
