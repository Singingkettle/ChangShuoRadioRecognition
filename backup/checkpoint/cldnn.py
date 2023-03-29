import torch

from csrr.common.utils import Config
from csrr.models import build_method
from csrr.runner.checkpoint import weights_to_cpu, get_state_dict

cfg = Config.fromfile('configs/cldnn/cldnn_iq-deepsig201610A.py')
model = build_method(cfg.model)


new_dict = weights_to_cpu(get_state_dict(model))
for key in list(new_dict.keys()):
       print(key)
       print(new_dict[key].shape)

checkpoint = torch.load('/home/citybuster/Data/SignalProcessing/Workdir/cldnn_iq-deepsig201610A/cldnn.pth',
                        map_location='cpu')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
state_dict = checkpoint['state_dict']
for key in list(state_dict.keys()):
       print(key)
       print(state_dict[key].shape)