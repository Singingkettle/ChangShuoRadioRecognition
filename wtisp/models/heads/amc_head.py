import torch
import torch.nn as nn

from .base_head import BaseHead
from ..builder import HEADS, build_loss, build_head


@HEADS.register_module()
class AMCHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, out_features=256,
                 loss_cls=None, requires_grad=True):
        super(AMCHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.loss_cls = build_loss(loss_cls)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.num_classes),
        )

        if not requires_grad:
            self.freeze_head()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def freeze_head(self):
        for m in self.parameters():
            m.requires_grad = False

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, vis_fea=False, **kwargs):
        x = x.reshape(-1, self.in_features)
        if vis_fea:
            x = self.classifier[0](x)
            fea = self.classifier[1](x)
            x = self.classifier[2](fea)
            pre = self.classifier[3](x)
            return dict(fea=fea, pre=pre)
        pre = self.classifier(x)

        return pre


@HEADS.register_module()
class DSAMCHead(BaseHead):
    def __init__(self, num_classes, in_features=2500, loss_cls=None):
        super(DSAMCHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.loss_cls = build_loss(loss_cls)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, self.in_features)
        pre = self.classifier(x)

        return pre


@HEADS.register_module()
class MergeAMCHead(BaseHead):
    def __init__(self, loss_cls=None):
        super(MergeAMCHead, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.loss_cls = build_loss(loss_cls)
        if loss_cls['type'] is 'NLLLoss':
            self.with_log = True
        else:
            self.with_log = False

    def init_weights(self):
        pass

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, **kwargs):
        snr_p = self.softmax(x['snr_pre'])
        low_p = self.softmax(x['low_pre'])
        high_p = self.softmax(x['high_pre'])

        low_snr_pred = torch.mul(low_p, snr_p[:, -1:])

        high_snr_pred = torch.mul(high_p, snr_p[:, :1])
        merge_pre = torch.add(low_snr_pred, high_snr_pred)

        if self.with_log:
            merge_pre = torch.where(merge_pre > 0, merge_pre, merge_pre.new_tensor(1))
            merge_pre = torch.log(merge_pre)

        return merge_pre


@HEADS.register_module()
class MLAMCHead(BaseHead):
    def __init__(self, heads):
        super(MLAMCHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, low_weight=None, high_weight=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['snr_pre'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['low_pre'], mod_labels=mod_labels, weight=low_weight)
        high_loss = self.classifier_head[2].loss(
            x['high_pre'], mod_labels=mod_labels, weight=high_weight)

        losses['loss_snr'] = snr_loss['loss_cls']
        losses['loss_low'] = low_loss['loss_cls']
        losses['loss_high'] = high_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        snr_pre = self.classifier_head[0](x)
        low_pre = self.classifier_head[1](x)
        high_pre = self.classifier_head[2](x)
        return dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre)


@HEADS.register_module()
class MLHead(BaseHead):
    def __init__(self, heads):
        super(MLHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, low_weight=None, high_weight=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['snr_pre'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['low_pre'], mod_labels=mod_labels, weight=low_weight)
        high_loss = self.classifier_head[2].loss(
            x['high_pre'], mod_labels=mod_labels, weight=high_weight)
        merge_loss = self.classifier_head[3].loss(
            x['merge_pre'], mod_labels=mod_labels)

        losses['loss_snr'] = snr_loss['loss_cls']
        losses['loss_low'] = low_loss['loss_cls']
        losses['loss_high'] = high_loss['loss_cls']
        losses['loss_merge'] = merge_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        snr_pre = self.classifier_head[0](x[0])
        low_pre = self.classifier_head[1](x[1])
        high_pre = self.classifier_head[2](x[2])
        x = dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre)
        merge_pre = self.classifier_head[3](x)

        return dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre, pre=merge_pre)


@HEADS.register_module()
class MLHeadNoWeight(BaseHead):
    def __init__(self, heads):
        super(MLHeadNoWeight, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['snr_pre'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['low_pre'], mod_labels=mod_labels)
        high_loss = self.classifier_head[2].loss(
            x['high_pre'], mod_labels=mod_labels)
        merge_loss = self.classifier_head[3].loss(
            x['merge_pre'], mod_labels=mod_labels)

        losses['loss_snr'] = snr_loss['loss_cls']
        losses['loss_low'] = low_loss['loss_cls']
        losses['loss_high'] = high_loss['loss_cls']
        losses['loss_merge'] = merge_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        snr_pre = self.classifier_head[0](x[0])
        low_pre = self.classifier_head[1](x[1])
        high_pre = self.classifier_head[2](x[2])
        x = dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre)
        merge_pre = self.classifier_head[3](x)

        return dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre, pre=merge_pre)


@HEADS.register_module()
class FMLHeadNoWeight(BaseHead):
    def __init__(self, heads):
        super(FMLHeadNoWeight, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['snr_pre'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['low_pre'], mod_labels=mod_labels)
        high_loss = self.classifier_head[2].loss(
            x['high_pre'], mod_labels=mod_labels)
        merge_loss = self.classifier_head[3].loss(
            x['merge_pre'], mod_labels=mod_labels)

        losses['loss_snr'] = snr_loss['loss_cls']
        losses['loss_low'] = low_loss['loss_cls']
        losses['loss_high'] = high_loss['loss_cls']
        losses['loss_merge'] = merge_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        snr_pre = self.classifier_head[0](x)
        low_pre = self.classifier_head[1](x)
        high_pre = self.classifier_head[2](x)
        x = dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre)
        merge_pre = self.classifier_head[3](x)

        return dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre, pre=merge_pre)


@HEADS.register_module()
class FMLHierarchicalHead(BaseHead):
    def __init__(self, heads):
        super(FMLHierarchicalHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['cnn_pre'], mod_labels=mod_labels)
        low_loss = self.classifier_head[1].loss(
            x['lstm1_pre'], mod_labels=mod_labels)
        high_loss = self.classifier_head[2].loss(
            x['lstm2_pre'], mod_labels=mod_labels)

        losses['loss_cnn'] = snr_loss['loss_cls']
        losses['loss_lstm1'] = low_loss['loss_cls']
        losses['loss_lstm2'] = high_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        cnn_pre = self.classifier_head[0](x['cnn'])
        lstm1_pre = self.classifier_head[1](x['lstm1'])
        lstm2_pre = self.classifier_head[2](x['lstm2'])

        return dict(cnn_pre=cnn_pre, lstm1_pre=lstm1_pre, lstm2_pre=lstm2_pre)


@HEADS.register_module()
class FMergeAMCHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, out_features=256, loss_cls=None):
        super(FMergeAMCHead, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.in_features = in_features
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.loss_cls = build_loss(loss_cls)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(out_features, num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x, mod_labels, weight=weight)
        return dict(loss_cls=loss_cls)

    def forward(self, x, **kwargs):
        f0_p = self.softmax(x['fx0'])
        f1_p = self.softmax(x['fx1'])
        f2_p = self.softmax(x['fx2'])

        wx = x['fx3'].reshape(-1, self.in_features)
        f3_p = self.softmax(self.classifier(wx))

        p = torch.stack([f0_p, f1_p, f2_p], dim=2)
        w = torch.unsqueeze(f3_p, dim=2)
        p = torch.bmm(p, w)
        fx3 = torch.squeeze(p)
        return fx3


@HEADS.register_module()
class FPNAMCHead(BaseHead):
    def __init__(self, heads):
        super(FPNAMCHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, **kwargs):
        losses = dict()
        fx0_loss = self.classifier_head[0].loss(
            x['fx0'], mod_labels=mod_labels)
        fx1_loss = self.classifier_head[1].loss(
            x['fx1'], mod_labels=mod_labels)
        fx2_loss = self.classifier_head[2].loss(
            x['fx2'], mod_labels=mod_labels)
        fx3_loss = self.classifier_head[3].loss(
            x['fx3'], mod_labels=mod_labels)

        losses['loss_fx0'] = fx0_loss['loss_cls']
        losses['loss_fx1'] = fx1_loss['loss_cls']
        losses['loss_fx2'] = fx2_loss['loss_cls']
        losses['loss_fx3'] = fx3_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        fx0_pre = self.classifier_head[0](x['fx0'])
        fx1_pre = self.classifier_head[1](x['fx1'])
        fx2_pre = self.classifier_head[2](x['fx2'])

        x = dict(fx0_pre=fx0_pre, fx1_pre=fx1_pre, fx2_pre=fx2_pre, fx3_pre=x['fx3'])
        fx3_pre = self.classifier_head[3](x)

        return dict(fx0_pre=fx0_pre, fx1_pre=fx1_pre, fx2_pre=fx2_pre, fx3_pre=fx3_pre)


@HEADS.register_module()
class FMLHead(BaseHead):
    def __init__(self, heads):
        super(FMLHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, low_weight=None, high_weight=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['snr_pre'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['low_pre'], mod_labels=mod_labels, weight=low_weight)
        high_loss = self.classifier_head[2].loss(
            x['high_pre'], mod_labels=mod_labels, weight=high_weight)
        merge_loss = self.classifier_head[3].loss(
            x['merge_pre'], mod_labels=mod_labels)

        losses['loss_snr'] = snr_loss['loss_cls']
        losses['loss_low'] = low_loss['loss_cls']
        losses['loss_high'] = high_loss['loss_cls']
        losses['loss_merge'] = merge_loss['loss_cls']

        return losses

    def forward(self, x, **kwargs):
        snr_pre = self.classifier_head[0](x)
        low_pre = self.classifier_head[1](x)
        high_pre = self.classifier_head[2](x)
        x = dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre)
        merge_pre = self.classifier_head[3](x)

        return dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre, pre=merge_pre)


@HEADS.register_module()
class FMLAUXHead(BaseHead):
    def __init__(self, heads):
        super(FMLAUXHead, self).__init__()
        self.num_head = len(heads)
        self.classifier_head = nn.ModuleList()
        for head in heads:
            head_block = build_head(head)
            self.classifier_head.append(head_block)

    def init_weights(self):
        for i in range(self.num_head):
            self.classifier_head[i].init_weights()

    def loss(self, x, mod_labels=None, snr_labels=None, low_weight=None, high_weight=None, **kwargs):
        losses = dict()
        snr_loss = self.classifier_head[0].loss(
            x['snr_pre'], mod_labels=snr_labels)
        low_loss = self.classifier_head[1].loss(
            x['low_pre'], mod_labels=mod_labels, weight=low_weight)
        high_loss = self.classifier_head[2].loss(
            x['high_pre'], mod_labels=mod_labels, weight=high_weight)
        merge_loss = self.classifier_head[3].loss(
            x['merge_pre'], mod_labels=mod_labels)
        inter_loss = self.classifier_head[4].loss(
            x['inter_pre']
        )
        intra_loss = self.classifier_head[5].loss(
            x['intra_pre'], mod_labels=mod_labels
        )

        losses['loss_snr'] = snr_loss['loss_cls']
        losses['loss_low'] = low_loss['loss_cls']
        losses['loss_high'] = high_loss['loss_cls']
        losses['loss_merge'] = merge_loss['loss_cls']
        losses['loss_inter'] = inter_loss['loss_inter_orthogonal']
        losses['loss_intra'] = intra_loss['loss_intra_orthogonal']

        return losses

    def forward(self, x, **kwargs):
        snr_pre = self.classifier_head[0](x[0])
        low_pre = self.classifier_head[1](x[1])
        high_pre = self.classifier_head[2](x[2])
        x_ = dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre)
        merge_pre = self.classifier_head[3](x_)

        return dict(snr_pre=snr_pre, low_pre=low_pre, high_pre=high_pre, pre=merge_pre, inter_pre=x[3],
                    intra_pre=x[3])


@HEADS.register_module()
class FAMCAUXHead(BaseHead):
    def __init__(self, num_classes, in_features=10560, batch_size=None,
                 out_features=256, loss_cls=None,
                 aux_head=None):
        super(FAMCAUXHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        if aux_head is None:
            dict(
                type='IntraOrthogonalHead',
                in_features=in_features,  # keep the same as snr head
                batch_size=batch_size,  # keep the same as samples_per_gpu
                num_classes=num_classes,
                mm='inner_product',
                is_abs=False,
                loss_aux=dict(
                    type='LogisticLoss',
                    loss_weight=1,
                    temperature=100,
                ),
            ),
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.loss_cls = build_loss(loss_cls)
        self.aux_head = build_head(aux_head)

        self.fea = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace=True),
        )
        self.pre = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, self.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x['pre'], mod_labels, weight=weight)
        loss_aux = self.aux_head.loss(x['fea'], mod_labels)
        return dict(loss_cls=loss_cls, loss_aux=loss_aux['loss_intra_orthogonal'])

    def forward(self, x, vis_fea=False, mode='train'):
        x = x.reshape(-1, self.in_features)
        fea = self.fea(x)
        pre = self.pre(fea)
        if (mode is 'train') or vis_fea:
            return dict(fea=fea, pre=pre)
        else:
            return pre


@HEADS.register_module()
class DSAMCAUXHead(BaseHead):
    def __init__(self, num_classes, in_features=2500, loss_cls=None,
                 batch_size=None, aux_head=None):
        super(DSAMCAUXHead, self).__init__()
        if loss_cls is None:
            loss_cls = dict(
                type='CrossEntropyLoss',
                multi_label=False,
            )
        self.num_classes = num_classes
        self.in_features = in_features
        self.loss_cls = build_loss(loss_cls)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.num_classes),
        )
        if aux_head is None:
            dict(
                type='IntraOrthogonalHead',
                in_features=in_features,  # keep the same as snr head
                batch_size=batch_size,  # keep the same as samples_per_gpu
                num_classes=num_classes,
                mm='inner_product',
                is_abs=False,
                loss_aux=dict(
                    type='LogisticLoss',
                    loss_weight=1,
                    temperature=100,
                ),
            ),
        self.aux_head = build_head(aux_head)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def loss(self, x, mod_labels=None, weight=None, **kwargs):
        loss_cls = self.loss_cls(x['pre'], mod_labels, weight=weight)
        loss_aux = self.aux_head.loss(x['fea'], mod_labels)
        return dict(loss_cls=loss_cls, loss_aux=loss_aux['loss_intra_orthogonal'])

    def forward(self, x, mode='train'):
        fea = x.reshape(-1, self.in_features)
        pre = self.classifier(x)

        if mode is 'train':
            return dict(fea=fea, pre=pre)
        else:
            return pre
