#
# @HEADS.register_module()
# class FastMLDNNHeadV2(BaseHead):
#     def __init__(self, num_classes, alpha=-0.0001, beta=1, in_size=10560, out_size=256, loss_cls=None, is_reg=True):
#         super(FastMLDNNHeadV2, self).__init__()
#         if loss_cls is None:
#             loss_cls = dict(
#                 type='CrossEntropyLoss',
#             )
#
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.in_size = in_size
#         self.out_size = out_size
#         self.loss_cls = build_loss(loss_cls)
#         self.is_reg = is_reg
#
#         self.fea = nn.Sequential(
#             nn.Linear(self.in_size, self.out_size),
#             nn.ReLU(inplace=True),
#         )
#         self.pre = nn.Sequential(
#             nn.Dropout(0.5),
#             Cosine(self.out_size, self.num_classes),
#         )
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)
#             if isinstance(m, Euclidean):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='relu')
#
#     def loss(self, inputs, targets, weight=None, snrs=None, **kwargs):
#         inputs = inputs * 64
#         margin = self.alpha * snrs + self.beta
#         a = torch.mul(inputs, margin.view(-1, 1))
#
#         mask = F.one_hot(targets['modulations'], num_classes=inputs.shape[1])
#         x = mask * a + (1 - mask) * inputs
#         loss_cls = self.loss_cls(x, targets['modulations'], weight=weight)
#
#         if self.is_reg:
#             p = sim_matrix(self.pre[1].weight, self.pre[1].weight) * 64
#             loss_expand = self.loss_cls(p, targets['modulations'].new_tensor(np.arange(inputs.shape[1])))
#             return dict(loss_cls=loss_cls, loss_expand=loss_expand)
#         else:
#             return dict(loss_cls=loss_cls)
#
#     def forward(self, x, vis_fea=False, is_test=False):
#         x = x.reshape(-1, self.in_size)
#         fea = self.fea(x)
#         pre = self.pre(fea)
#         if is_test:
#             pre = torch.softmax(pre, dim=1)
#             if vis_fea:
#                 return dict(fea=fea, pre=pre, center=unit_vector(self.pre[1].weight))
#             else:
#                 return pre
#         else:
#             return pre
#
#
# @HEADS.register_module()
# class FastMLDNNHeadv3(BaseHead):
#     def __init__(self, num_classes, alpha=-0.0001, beta=1, in_size=10560, out_size=256, loss_cls=None, is_reg=True):
#         super(FastMLDNNHeadv3, self).__init__()
#         if loss_cls is None:
#             loss_cls = dict(
#                 type='CrossEntropyLoss',
#             )
#
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.in_size = in_size
#         self.out_size = out_size
#         self.loss_cls = build_loss(loss_cls)
#         self.is_reg = is_reg
#
#         self.fea = nn.Sequential(
#             nn.Linear(self.in_size, self.out_size),
#             nn.ReLU(inplace=True),
#         )
#         self.pre = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(self.out_size, self.num_classes),
#         )
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)
#             if isinstance(m, Euclidean):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='relu')
#
#     def loss(self, inputs, targets, weight=None, snrs=None, **kwargs):
#         # margin = self.beta * torch.exp(self.alpha - snrs)
#         margin = self.alpha * snrs + self.beta
#         a = torch.mul(inputs, margin.view(-1, 1))
#
#         mask = F.one_hot(targets['modulations'], num_classes=inputs.shape[1])
#         x = mask * a + (1 - mask) * inputs
#         loss_cls = self.loss_cls(x, targets['modulations'], weight=weight)
#
#         if self.is_reg:
#             p = torch.mm(self.pre[1].weight, self.pre[1].weight.transpose(0, 1)) * 0.1
#             loss_expand = self.loss_cls(-1 * p, targets['modulations'].new_tensor(np.arange(inputs.shape[1])))
#             return dict(loss_cls=loss_cls, loss_expand=loss_expand)
#         else:
#             return dict(loss_cls=loss_cls)
#
#     def forward(self, x, vis_fea=False, is_test=False):
#         x = x.reshape(-1, self.in_size)
#         fea = self.fea(x)
#         pre = self.pre(fea)
#         if vis_fea:
#             if is_test:
#                 pre = torch.softmax(pre, dim=1)
#             return dict(fea=fea, pre=pre, center=self.pre[1].weight)
#         else:
#             if is_test:
#                 pre = torch.softmax(pre, dim=1)
#             return pre


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .base_head import BaseHead
# from ..builder import HEADS, build_loss
# from ...runner import Sequential
#
#
# @HEADS.register_module()
# class FastMLDNNHead(BaseHead):
#     def __init__(self, num_classes, alpha=-0.0001, beta=1, in_size=10560,
#                  out_size=256, loss_cls=None, loss_rl_cls=None, is_reg=True, init_cfg=None):
#         super(FastMLDNNHead, self).__init__(init_cfg)
#         if loss_cls is None:
#             loss_cls = dict(
#                 type='CrossEntropyLoss',
#             )
#         if loss_rl_cls is None:
#             loss_rl_cls = dict(
#                 type='CrossEntropyLoss',
#             )
#
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.in_size = in_size
#         self.out_size = out_size
#         self.loss_cls = build_loss(loss_cls)
#         self.loss_rl_cls = build_loss(loss_rl_cls)
#         self.is_reg = is_reg
#
#         self.fea = Sequential(
#             nn.Linear(self.in_size, self.out_size),
#             nn.ReLU(inplace=True),
#         )
#         self.pre = Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(self.out_size, self.num_classes),
#         )
#
#     def loss(self, inputs, targets, weight=None, **kwargs):
#         loss_rl_cls = self.loss_rl_cls(inputs, targets['rl_modulations'], weight=weight)
#         loss_cls = self.loss_cls(inputs, targets['modulations'], weight=weight)
#         return dict(loss_cls=loss_cls, loss_rl_cls=loss_rl_cls)
#
#     def forward(self, x, vis_fea=False, is_test=False):
#         x = x.reshape(-1, self.in_size)
#         fea = self.fea(x)
#         pre = self.pre(fea)
#         if vis_fea:
#             if is_test:
#                 pre = torch.softmax(pre, dim=1)
#             return dict(fea=fea, pre=pre, center=self.pre[1].weight)
#         else:
#             if is_test:
#                 pre = torch.softmax(pre, dim=1)
#             return pre


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .base_head import BaseHead
# from ..builder import HEADS, build_loss
# from ...runner import Sequential
#
#
# @HEADS.register_module()
# class FastMLDNNHead(BaseHead):
#     def __init__(self, num_classes, alpha=-0.0001, beta=1, in_size=10560,
#                  out_size=256, loss_cls=None, loss_circle=None, is_reg=True, init_cfg=None):
#         super(FastMLDNNHead, self).__init__(init_cfg)
#         if loss_cls is None:
#             loss_cls = dict(
#                 type='CrossEntropyLoss',
#             )
#         if loss_circle is None:
#             loss_circle = dict(
#                 type='CircleLoss',
#             )
#
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.in_size = in_size
#         self.out_size = out_size
#         self.loss_cls = build_loss(loss_cls)
#         self.loss_circle = build_loss(loss_circle)
#         self.is_reg = is_reg
#
#         self.fea = Sequential(
#             nn.Linear(self.in_size, self.out_size),
#             nn.ReLU(inplace=True),
#         )
#         self.pre = Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(self.out_size, self.num_classes),
#         )
#
#     def loss(self, inputs, targets, weight=None, **kwargs):
#         loss_circle = self.loss_circle(inputs['fea'], targets['modulations'], weight=weight)
#         loss_cls = self.loss_cls(inputs['pre'], targets['modulations'], weight=weight)
#         return dict(loss_cls=loss_cls, loss_circle=loss_circle)
#
#     def forward(self, x, vis_fea=True, is_test=False):
#         x = x.reshape(-1, self.in_size)
#         fea = self.fea(x)
#         pre = self.pre(fea)
#         if vis_fea:
#             if is_test:
#                 pre = torch.softmax(pre, dim=1)
#             return dict(fea=fea, pre=pre)
#         else:
#             if is_test:
#                 pre = torch.softmax(pre, dim=1)
#             return pre
