from .alexnet_co import AlexNetCO
from .cemee import CEMEENet
from .cnnnet import CNNNet, ResNet, DensNet
from .cobonet import CoBoNet
from .cpcstn import (CPCNN, CSTN, CPCSTN)
from .crnet import CRNet
from .ctnet import CTNet, CTNetV2
from .dsclnet import DSCLNet
from .fmlnet import (FMLNetV1, FMLNetV2, FMLNetV3, FMLNetV4, FMLNetV5, FMLNetV6, FMLNetV7, FMLNetV8, FMLNetV9,
                     FMLNetV10, FMLNetV11, FMLNetV12, FMLNetV13, FMLNetV14, FMLNetV15, FMLNetV16, FMLNetV17,
                     FMLNetV18, FMLNetV19, FMLNetV20, FMLNetV21, FMLNetV22, FMLNetV23, FMLNetV24, FMLNetV25,
                     FMLNetV26, FMLNetV27, FMLNetV28, FMLNetV29, FMLNetV30, FMLNetV31, FMLNetV32, FMLNetV33,
                     FMLNetV34, FMLNetV35, FMLNetV36, FMLNetV37, FMLNetV38, FMLNetV39, FMLNetV40, FMLNetV41,
                     FMLNetV42, FMLNetV43, FMLNetV44, FMLNetV45, FMLNetV46, FMLNetV47, FMLNetV48, FMLNetV49)
from .googlenet_co import GoogleNetCO
from .hcgnet import (HCGNetV1, HCGNetV2, HCGNetV3, HCGNetV4, HCGNetV5, HCGNetCNN, HCGNetGRU1, HCGNetGRU2,
                     HCGNetCG1, HCGNetCG2, HCGNetG1G2)
from .mctnet import MCTNetV1, MCTNetV2
from .mlclnet import MLCLNet
from .mlnet import (MLNetV1, MLNetV2, MLNetV3, MLNetV4, MLNetV5,
                    MLNetV6, MLNetV7, MLNetV8, MLNetV9, MLNetV10,
                    MLNetV11, MLNetV12)
from .resclnet import ResCLNetV1
from .resnet_co import ResNetCO
from .tanhnet import TanhNet
from .vggnet_co import VGGNetCO

__all__ = [
    'CNNNet', 'ResNet', 'DensNet',
    'TanhNet', 'CRNet', 'DSCLNet',
    'CTNet', 'CTNetV2', 'MLCLNet',
    'MLNetV1', 'MLNetV2', 'MLNetV3',
    'MLNetV4', 'MLNetV5', 'MLNetV6',
    'MLNetV7', 'MLNetV8', 'MLNetV9',
    'MLNetV10', 'MLNetV11', 'MLNetV12',
    'HCGNetV1', 'FMLNetV1', 'FMLNetV2',
    'FMLNetV3', 'FMLNetV4', 'FMLNetV4',
    'FMLNetV6', 'FMLNetV7', 'FMLNetV8',
    'FMLNetV9', 'FMLNetV10', 'FMLNetV11',
    'FMLNetV12', 'FMLNetV13', 'FMLNetV14',
    'FMLNetV15', 'FMLNetV16', 'FMLNetV17',
    'FMLNetV5', 'FMLNetV18', 'FMLNetV19',
    'FMLNetV20', 'FMLNetV21', 'FMLNetV22',
    'FMLNetV23', 'FMLNetV24', 'FMLNetV25',
    'FMLNetV26', 'FMLNetV30', 'FMLNetV33',
    'FMLNetV27', 'FMLNetV28', 'FMLNetV29',
    'AlexNetCO', 'GoogleNetCO', 'ResNetCO',
    'VGGNetCO', 'CoBoNet', 'FMLNetV34',
    'FMLNetV36', 'FMLNetV37', 'FMLNetV38',
    'FMLNetV39', 'FMLNetV40', 'FMLNetV41',
    'FMLNetV42', 'FMLNetV44', 'FMLNetV43',
    'FMLNetV45', 'FMLNetV46', 'FMLNetV47',
    'FMLNetV48', 'FMLNetV35', 'FMLNetV49',
    'FMLNetV35', 'FMLNetV32', 'FMLNetV31',
    'CPCNN', 'CSTN', 'CPCSTN', 'ResCLNetV1',
    'CEMEENet', 'MCTNetV1', 'MCTNetV2',
    'HCGNetV3', 'HCGNetV2', 'HCGNetV4', 'HCGNetV5',
    'HCGNetCNN', 'HCGNetGRU1', 'HCGNetGRU2',
    'HCGNetCG1', 'HCGNetCG2', 'HCGNetG1G2'
]
