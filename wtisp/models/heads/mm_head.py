import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class ASDHead(BaseHead):
    def __init__(self, amc_cls_num, sei_cls_num, in_features=256, out_features=256,
                 loss_amc=None, loss_sei=None):
        super(ASDHead, self).__init__()
        self.amc_cls_num = amc_cls_num
        self.sei_cls_num = sei_cls_num
        self.amc_classifier = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(out_features, amc_cls_num)
        )
        self.sei_classifier = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(out_features, sei_cls_num)
        )
        if loss_amc is None:
            loss_amc = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )

        if loss_sei is None:
            loss_sei = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )

        self.loss_amc = build_loss(loss_amc)
        self.loss_sei = build_loss(loss_sei)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, dev_labels=None):
        losses = dict()

        sei_loss = self.loss_sei(x['SEI'], dev_labels)
        amc_loss = self.loss_amc(x['AMC'], mod_labels)

        losses['loss_SEI'] = sei_loss
        losses['loss_AMC'] = amc_loss

        return losses

    def forward(self, x, vis_fea=False, is_test=False):
        amc = self.amc_classifier(x)  # B * K1 * 1 * 1
        sei = self.sei_classifier(x)

        return dict(AMC=amc, SEI=sei)


class MMHead(nn.Module):
    def __init__(self, main_cls_num, minor_cls_num, in_features=256, out_features=256, is_share=False):
        super(MMHead, self).__init__()
        self.main_classifier = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(out_features, main_cls_num, kernel_size=1)
        )
        if is_share:
            self.minor_classifier = nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(out_features, main_cls_num * minor_cls_num, kernel_size=1)
            )
        else:
            self.minor_classifier = nn.Sequential(
                nn.Conv1d(in_features, out_features * main_cls_num, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(out_features * main_cls_num, main_cls_num * minor_cls_num, kernel_size=1, groups=main_cls_num)
            )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        main_pre = self.main_classifier(x)
        minor_pre = self.minor_classifier(x)
        main_pre = torch.squeeze(main_pre)
        minor_pre = torch.squeeze(minor_pre)

        return dict(Main=main_pre, Minor=minor_pre)


@HEADS.register_module()
class ASSHead(BaseHead):
    def __init__(self, amc_cls_num, sei_cls_num, in_features=256, out_features=256,
                 is_abs=True, loss_amc=None, loss_sei=None, is_share=False):
        super(ASSHead, self).__init__()
        self.amc_cls_num = amc_cls_num
        self.sei_cls_num = sei_cls_num
        self.is_abs = is_abs
        if self.is_abs:
            self.classifier = MMHead(sei_cls_num, amc_cls_num, in_features, out_features, is_share)
        else:
            self.classifier = MMHead(amc_cls_num, sei_cls_num, in_features, out_features, is_share)

        if loss_amc is None:
            loss_amc = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )

        if loss_sei is None:
            loss_sei = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )

        self.loss_amc = build_loss(loss_amc)
        self.loss_sei = build_loss(loss_sei)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, dev_labels=None, mask_weight=None):
        losses = dict()
        if self.is_abs:
            amc = torch.reshape(x['AMC'], [-1, self.sei_cls_num, self.amc_cls_num])
            amc = amc * mask_weight
            amc = torch.sum(amc, dim=1)
            sei_loss = self.loss_sei(x['SEI'], dev_labels)
            amc_loss = self.loss_amc(amc, mod_labels)
        else:
            sei = torch.reshape(x['SEI'], [-1, self.amc_cls_num, self.sei_cls_num])
            sei = sei * mask_weight
            sei = torch.sum(sei, dim=1)
            sei_loss = self.loss_sei(sei, dev_labels)
            amc_loss = self.loss_amc(x['AMC'], mod_labels)

        losses['loss_SEI'] = sei_loss
        losses['loss_AMC'] = amc_loss

        return losses

    def forward(self, x, vis_fea=False, is_test=False):
        pre = self.classifier(x)  # B * K1 * 1 * 1
        if is_test:
            main_pre = pre['Main']
            minor_pre = pre['Minor']
            bn, k = main_pre.shape
            if self.is_abs:
                minor_pre = torch.reshape(minor_pre, [-1, self.sei_cls_num, self.amc_cls_num])
            else:
                minor_pre = torch.reshape(minor_pre, [-1, self.amc_cls_num, self.sei_cls_num])
            save_index = torch.argmax(main_pre, dim=1)
            mask = x.new_full((bn, k, 1), 0)
            mask[torch.arange(bn), save_index, 0] = 1
            minor_pre = minor_pre * mask
            minor_pre = torch.sum(minor_pre, dim=1)
            pre['Minor'] = minor_pre
        if self.is_abs:
            return dict(AMC=pre['Minor'], SEI=pre['Main'])
        else:
            return dict(AMC=pre['Main'], SEI=pre['Minor'])
