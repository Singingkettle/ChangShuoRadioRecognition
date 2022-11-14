import torch
import torch.nn as nn

from .amc_head import AMCHead
from .base_head import BaseHead
from .mm_head import MMHead
from ..builder import HEADS, build_loss, build_head


@HEADS.register_module()
class GBBCEHead(AMCHead):
    def __init__(self, num_classes, in_features=10560, out_features=256, loss_cls=None):
        super(GBBCEHead, self).__init__(num_classes, in_features, out_features, loss_cls)

    def forward(self, x, vis_fea=False, is_test=False):
        x = x.reshape(-1, self.in_features)
        x = self.classifier(x)
        if is_test:
            x = torch.sigmoid(x)
        return x


@HEADS.register_module()
class GBIndHead(BaseHead):
    def __init__(self, channel_head, mod_head, order_head):
        super(GBIndHead, self).__init__()
        self.channel_head = build_head(channel_head)
        self.mod_head = build_head(mod_head)
        self.order_head = build_head(order_head)

    def init_weights(self):
        self.channel_head.init_weights()
        self.mod_head.init_weights()
        self.order_head.init_weights()

    def loss(self, x, mod_labels=None, channel_labels=None, order_labels=None):
        channel_loss = self.channel_head.loss(x['channel'], channel_labels)['loss_Final']
        mod_loss = self.mod_head.loss(x['mod'], mod_labels)['loss_Final']
        order_loss = self.order_head.loss(x['order'], order_labels)['loss_Final']
        return dict(loss_Channel=channel_loss, loss_Mod=mod_loss, loss_Order=order_loss)

    def forward(self, x, vis_fea=False, is_test=False):
        channel_pre = self.channel_head(x)
        mod_pre = self.mod_head(x)
        order_pre = self.order_head(x)
        if is_test:
            channel_pre = torch.sigmoid(channel_pre)
            mod_pre = torch.sigmoid(mod_pre)
            order_pre = torch.sigmoid(order_pre)

        return dict(channel=channel_pre, mod=mod_pre, order=order_pre)


@HEADS.register_module()
class GBDetHead(BaseHead):
    def __init__(self, channel_cls_num, mod_cls_num, in_features=256, out_features=256,
                 is_share=False, loss_channel=None, loss_mod=None):
        super(GBDetHead, self).__init__()
        self.channel_cls_num = channel_cls_num
        self.mod_cls_num = mod_cls_num
        self.classifier = MMHead(channel_cls_num, mod_cls_num, in_features,
                                 out_features, is_share, main_name='Channel', minor_name='Mod')
        self.loss_channel = build_loss(loss_channel)
        self.loss_mod = build_loss(loss_mod)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, channel_labels=None, mod_labels=None):
        channel_loss = self.loss_channel(x['Channel'], channel_labels)
        mod_loss = self.loss_mod(x['Mod'], mod_labels)

        return dict(loss_channel=channel_loss, loss_mod=mod_loss)

    def forward(self, x, vis_fea=False, is_test=False):
        x = self.classifier(x)
        if is_test:
            for key_str in x:
                x[key_str] = torch.sigmoid(x[key_str])

        return x
