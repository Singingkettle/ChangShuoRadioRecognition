new = ['backbone.iq_net.conv_net.0.weight', 'backbone.iq_net.conv_net.0.bias',
       'backbone.iq_net.conv_net.3.weight', 'backbone.iq_net.conv_net.3.bias',
       'backbone.iq_net.conv_net.6.weight', 'backbone.iq_net.conv_net.6.bias',
       'backbone.iq_net.fusion.fc.0.weight', 'backbone.iq_net.fusion.fc.2.weight',

       'backbone.iq_net.gru_net.weight_ih_l0', 'backbone.iq_net.gru_net.weight_hh_l0',
       'backbone.iq_net.gru_net.bias_ih_l0', 'backbone.iq_net.gru_net.bias_hh_l0',
       'backbone.iq_net.gru_net.weight_ih_l0_reverse', 'backbone.iq_net.gru_net.weight_hh_l0_reverse',
       'backbone.iq_net.gru_net.bias_ih_l0_reverse', 'backbone.iq_net.gru_net.bias_hh_l0_reverse',

       'backbone.iq_net.gru_net.weight_ih_l1', 'backbone.iq_net.gru_net.weight_hh_l1',
       'backbone.iq_net.gru_net.bias_ih_l1', 'backbone.iq_net.gru_net.bias_hh_l1',
       'backbone.iq_net.gru_net.weight_ih_l1_reverse', 'backbone.iq_net.gru_net.weight_hh_l1_reverse',
       'backbone.iq_net.gru_net.bias_ih_l1_reverse', 'backbone.iq_net.gru_net.bias_hh_l1_reverse',

       'backbone.ap_net.conv_net.0.weight', 'backbone.ap_net.conv_net.0.bias',
       'backbone.ap_net.conv_net.3.weight', 'backbone.ap_net.conv_net.3.bias',
       'backbone.ap_net.conv_net.6.weight', 'backbone.ap_net.conv_net.6.bias',
       'backbone.ap_net.fusion.fc.0.weight', 'backbone.ap_net.fusion.fc.2.weight',

       'backbone.ap_net.gru_net.weight_ih_l0', 'backbone.ap_net.gru_net.weight_hh_l0',
       'backbone.ap_net.gru_net.bias_ih_l0', 'backbone.ap_net.gru_net.bias_hh_l0',
       'backbone.ap_net.gru_net.weight_ih_l0_reverse', 'backbone.ap_net.gru_net.weight_hh_l0_reverse',
       'backbone.ap_net.gru_net.bias_ih_l0_reverse', 'backbone.ap_net.gru_net.bias_hh_l0_reverse',

       'backbone.ap_net.gru_net.weight_ih_l1', 'backbone.ap_net.gru_net.weight_hh_l1',
       'backbone.ap_net.gru_net.bias_ih_l1', 'backbone.ap_net.gru_net.bias_hh_l1',
       'backbone.ap_net.gru_net.weight_ih_l1_reverse', 'backbone.ap_net.gru_net.weight_hh_l1_reverse',
       'backbone.ap_net.gru_net.bias_ih_l1_reverse', 'backbone.ap_net.gru_net.bias_hh_l1_reverse',

       'classifier_head.snr.classifier.0.weight', 'classifier_head.snr.classifier.0.bias',
       'classifier_head.snr.classifier.3.weight', 'classifier_head.snr.classifier.3.bias',

       'classifier_head.low.classifier.0.weight', 'classifier_head.low.classifier.0.bias',
       'classifier_head.low.classifier.3.weight', 'classifier_head.low.classifier.3.bias',

       'classifier_head.high.classifier.0.weight', 'classifier_head.high.classifier.0.bias',
       'classifier_head.high.classifier.3.weight', 'classifier_head.high.classifier.3.bias']

old = ['backbone.iq_conv_net.0.weight', 'backbone.iq_conv_net.0.bias',
       'backbone.iq_conv_net.3.weight', 'backbone.iq_conv_net.3.bias',
       'backbone.iq_conv_net.6.weight', 'backbone.iq_conv_net.6.bias',
       'backbone.iq_se.fc.0.weight', 'backbone.iq_se.fc.2.weight',

       'backbone.iq_gru.weight_ih_l0', 'backbone.iq_gru.weight_hh_l0',
       'backbone.iq_gru.bias_ih_l0', 'backbone.iq_gru.bias_hh_l0',
       'backbone.iq_gru.weight_ih_l0_reverse', 'backbone.iq_gru.weight_hh_l0_reverse',
       'backbone.iq_gru.bias_ih_l0_reverse', 'backbone.iq_gru.bias_hh_l0_reverse',

       'backbone.iq_gru.weight_ih_l1', 'backbone.iq_gru.weight_hh_l1',
       'backbone.iq_gru.bias_ih_l1', 'backbone.iq_gru.bias_hh_l1',
       'backbone.iq_gru.weight_ih_l1_reverse', 'backbone.iq_gru.weight_hh_l1_reverse',
       'backbone.iq_gru.bias_ih_l1_reverse', 'backbone.iq_gru.bias_hh_l1_reverse',

       'backbone.ap_conv_net.0.weight', 'backbone.ap_conv_net.0.bias',
       'backbone.ap_conv_net.3.weight', 'backbone.ap_conv_net.3.bias',
       'backbone.ap_conv_net.6.weight', 'backbone.ap_conv_net.6.bias',
       'backbone.ap_se.fc.0.weight', 'backbone.ap_se.fc.2.weight',

       'backbone.ap_gru.weight_ih_l0', 'backbone.ap_gru.weight_hh_l0',
       'backbone.ap_gru.bias_ih_l0', 'backbone.ap_gru.bias_hh_l0',
       'backbone.ap_gru.weight_ih_l0_reverse', 'backbone.ap_gru.weight_hh_l0_reverse',
       'backbone.ap_gru.bias_ih_l0_reverse', 'backbone.ap_gru.bias_hh_l0_reverse',

       'backbone.ap_gru.weight_ih_l1', 'backbone.ap_gru.weight_hh_l1',
       'backbone.ap_gru.bias_ih_l1', 'backbone.ap_gru.bias_hh_l1',
       'backbone.ap_gru.weight_ih_l1_reverse', 'backbone.ap_gru.weight_hh_l1_reverse',
       'backbone.ap_gru.bias_ih_l1_reverse', 'backbone.ap_gru.bias_hh_l1_reverse',

       'classifier_head.classifier_head.0.classifier.0.weight', 'classifier_head.classifier_head.0.classifier.0.bias',
       'classifier_head.classifier_head.0.classifier.3.weight', 'classifier_head.classifier_head.0.classifier.3.bias',

       'classifier_head.classifier_head.1.classifier.0.weight', 'classifier_head.classifier_head.1.classifier.0.bias',
       'classifier_head.classifier_head.1.classifier.3.weight', 'classifier_head.classifier_head.1.classifier.3.bias',

       'classifier_head.classifier_head.2.classifier.0.weight', 'classifier_head.classifier_head.2.classifier.0.bias',
       'classifier_head.classifier_head.2.classifier.3.weight', 'classifier_head.classifier_head.2.classifier.3.bias']

import torch

from csrr.common.utils import Config
from csrr.models import build_method
from csrr.runner.checkpoint import weights_to_cpu, get_state_dict

cfg = Config.fromfile('configs/mldnn/mldnn_iq-ap-deepsig201801A.py')
model = build_method(cfg.model)

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
new_dict = weights_to_cpu(get_state_dict(model))
for key in list(new_dict.keys()):
       print(key)
       print(new_dict[key].shape)

checkpoint = torch.load('work_dirs/mldnn_iq-ap-deepsig201801A/epoch_380.pth',
                        map_location='cpu')

state_dict = checkpoint['state_dict']
for key in list(state_dict.keys()):
       print(key)
       print(state_dict[key].shape)


rename = dict()
for i, name in enumerate(old):
       print('@@@@@@@@@@@@@@@@@@@@@@@@')
       print(i)
       print(name)
       print(new[i])
       rename[name] = new[i]

print(rename)

print('*****************************************************************************************************')
for key in list(state_dict.keys()):
       print(key)
       print(checkpoint['state_dict'][key].shape)
       checkpoint['state_dict'][rename[key]] = checkpoint['state_dict'][key]
       del checkpoint['state_dict'][key]


with open('work_dirs/mldnn_iq-ap-deepsig201801A/mldnn.pth', 'wb') as f:
       torch.save(checkpoint, f)
       f.flush()


checkpoint = torch.load('work_dirs/mldnn_iq-ap-deepsig201801A/mldnn.pth',
                        map_location='cpu')

state_dict = checkpoint['state_dict']

print('==============================================================================')
for key in list(state_dict.keys()):
       print(key)
       print(checkpoint['state_dict'][key].shape)