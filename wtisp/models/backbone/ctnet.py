import logging
import math

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ...runner import load_checkpoint


##### 这个类与旧版一致
## 每个attention layer最后一步都是经过一个FF层来让特征更加丰富
class FeedForward(nn.Module):
    def __init__(self, seq_dim, ff_hidden_dim, ff_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_dim, ff_hidden_dim),
            nn.GELU(),  # 高斯误差激活函数，形状类似于ReLU，但是拐角处更加平滑
            nn.Dropout(ff_dropout),
            nn.Linear(ff_hidden_dim, seq_dim),
            nn.Dropout(ff_dropout)
        )

    def forward(self, x):
        return self.net(x)


# 照搬 "http://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html"的实现方法

##### 这个类与旧版一致
# 对输入的q,k,v张量进行维度变换，保证后续计算注意力的正确性
# 以X代指输入的张量，其维度是(batch,seq_len,num_hiddens)
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads,
                  -1)  # 按照多头数量拆分，维度变换至(batch,seq_len,num_heads,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)  # 一次性转换维度至(batch,num_heads,seq_len,num_hiddens/num_heads)
    X = X.reshape(-1, X.shape[2], X.shape[3])  # 把batch和num_heads合并成新的batch，维度变为(batch*num_heads,seq_len,new_dim)
    return X


##### 这个类与旧版一致
# 在计算完多头注意力机制后，转变维度变成正常的维度顺序
# 输入的维度是(batch*num_heads,seq_len,new_dim)
def transpose_out(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])  # (batch,num_heads,seq_len,new_dim)
    X = X.permute(0, 2, 1, 3)  # (batch,seq_len,num_heads,new_dim)
    X = X.reshape(X.shape[0], X.shape[1], -1)  # (batch,seq_len,num_heads*new_dim)  这个操作相当于是把多头的计>算张量，直接首尾拼接在一起
    return X


##### 这个类与旧版一致
class DotProductAttention(nn.Module):
    def __init__(self, attn_weight_dropout):  # 这个注意力层也可以添加dropout操作
        super().__init__()
        self.dropout = nn.Dropout(attn_weight_dropout)

    # 输入的q,k,v必须是同样维度的,都是(batch*num_head,seq_len,new_dim)
    def forward(self, queries, keys, values):
        dk = queries.shape[-1]  # 注意力机制中的缩放因子
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(dk)  # 维度是(batch*num_head,seq_len,seq_len)

        self.attention_weights = scores.softmax(dim=-1)  # 把q*k的得分转换成概率, 维度是(batch*num_head,seq_len,seq_len)
        self.real_attn = self.dropout(self.attention_weights)  # 维度是(batch*num_head,seq_len,seq_len)
        out = torch.bmm(self.real_attn, values)  # 维度是(batch*num_head,seq_len,new_dim)
        return out


##### 这个类与旧版一致
class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, attn_weight_dropout, bias, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)  # 之前的版本使用的是False设置，现在改成了True，估计影响>不大
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_out = nn.Linear(num_hiddens, num_hiddens, bias=bias)

        self.attention = DotProductAttention(attn_weight_dropout)

    # 这个模块要求输入三个张量，分别是q,k,v，他们的维度均是(batch,seq_len,seq_dim)，在大多数场景下，query=key=value=X
    def forward(self, in_query, in_key, in_value):
        tmp_query = self.W_q(in_query)  # (batch,seq_len,seq_dim) -> (batch,seq_len,num_hiddens)
        out_query = transpose_qkv(tmp_query,
                                  self.num_heads)  # (batch,seq_len,seq_dim) -> (batch*num_heads,seq_len,new_dim)

        tmp_key = self.W_k(in_key)
        out_key = transpose_qkv(tmp_key, self.num_heads)

        tmp_value = self.W_v(in_value)
        out_value = transpose_qkv(tmp_value, self.num_heads)

        attn_out = self.attention(out_query, out_key, out_value)  # 输出维度 (batch*num_head,seq_len,new_dim)
        out = transpose_out(attn_out, self.num_heads)  # 输出维度 (batch,seq_len,num_hiddens)  num_hiddens=num_head*new_dim
        out = self.W_out(out)

        return out


## 对比了一下旧版，也没啥区别，参数设置也相同
## 注意力层（在标准TRM中，叫做Encoder层）
## 由残差连接、LN规范化、多头注意力MHA、前向反馈计算FF组合而来
## 这一块的组合原理可能还需要仔细琢磨琢磨
class Atten_Layer(nn.Module):
    def __init__(self, seq_dim, heads, ff_hidden_dim, attn_weight_dropout, ff_dropout, dropout_after_attn):
        super().__init__()

        self.first_LN = nn.LayerNorm(seq_dim)  # 根据大多数TRM的研究结论，添加一个层规范化操作（但是我觉得在AMC领域中>，有没有可能BN效果更好一些）
        self.second_LN = nn.LayerNorm(seq_dim)  # 根据Transformer的原文，一个encoder中有两个LayerNorm层

        self.attention = MultiHeadAttention(seq_dim, seq_dim, seq_dim, seq_dim, heads, attn_weight_dropout, False)
        self.ff_layer = FeedForward(seq_dim, ff_hidden_dim, ff_dropout)  # mlp_dim表示中间隐藏层的神经元数量，不影响FeedForward最终输出向量长度
        self.droplayer_after_attn = nn.Dropout(dropout_after_attn)

    def forward(self, inputs):
        # inputs的维度是(batch,length,dimension)
        x = self.first_LN(inputs)  # 输出维度：(batch,length,dimension)
        x = self.attention(x, x, x)  # 输出维度：(batch,length,dimension)
        x = self.droplayer_after_attn(x)  # 输出维度：(batch,length,dimension)
        x = x + inputs  # 残差连接

        y = self.second_LN(x)  # 输出维度：(batch,length,dimension)
        y = self.ff_layer(y)  # 输出维度：(batch,length,dimension)
        return x + y  # 去掉了这一层残差链接效果会变差，感觉残差要无孔不入的添加才有效


# 这一版densenet 可以调节深度和宽度，但是默认把输入的通道也会添加到输出中去，没使用maxpool操作
# 这里使用bn,relu,conv的顺序,是否需要调整一下顺序
def conv_block(in_channel, out_channel, kernel_size):
    padd = int((kernel_size - 1) / 2)
    layer = nn.Sequential(
        nn.BatchNorm1d(in_channel),
        nn.ReLU(True),
        nn.Conv1d(in_channel, out_channel, kernel_size, padding=padd, bias=False)
    )
    return layer


@BACKBONES.register_module()
class CTNet(nn.Module):
    # created by YSC, use the parameters as CTDNN
    # This version of densenet cascades each layer
    def __init__(self, in_channel, growth_rate, num_layers, kernel_size):
        super(CTNet, self).__init__()
        block = []
        channel = in_channel

        # add base conv block to Densenet
        for i in range(num_layers):
            block.append(self.base_conv_block(channel, growth_rate, kernel_size))
            channel += growth_rate

        self.net = nn.Sequential(*block)
        self.flat = nn.Flatten()

        seq_dim = 2 + growth_rate * num_layers
        heads = 2
        seq_len = 256 + 1
        ff_hidden_dim = 1024
        attn_weight_dropout = 0.4
        ff_dropout = 0.1
        dropout_after_attn = 0.4

        self.cls = nn.Parameter(torch.randn(1, 1, seq_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, seq_dim))  # 动态学习的PE编码方案
        self.all_attention_layers = nn.ModuleList([])
        self.all_attention_layers.append(
            Atten_Layer(seq_dim, heads, ff_hidden_dim, attn_weight_dropout, ff_dropout, dropout_after_attn))
        self.all_attention_layers.append(
            Atten_Layer(seq_dim, heads, ff_hidden_dim, attn_weight_dropout, ff_dropout, dropout_after_attn))

        self.final_layernorm = nn.LayerNorm(seq_dim)

    def base_conv_block(self, in_channel, out_channel, kernel_size):
        padd = int((kernel_size - 1) / 2)  # Ensure that the length of the input and output remains the same
        layer = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.ReLU(True),
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=padd, bias=False)
        )
        return layer

    def init_weights(self, pre_trained=None):
        if isinstance(pre_trained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pre_trained, strict=False, logger=logger)
        elif pre_trained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, iqs):
        x = iqs[:, :, :]
        for n, layer in enumerate(self.net):
            # print(x.shape)
            out = layer(x)
            x = torch.cat((out, x), dim=1)

        # x = self.flat(x)

        batch_size = x.shape[0]
        cnn_x = torch.transpose(x, 1, 2)
        cls_tokens = self.cls.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, cnn_x), dim=1)
        x += self.pos_embedding[:, :]

        for layer in self.all_attention_layers:
            x = layer(x)

        x = self.final_layernorm(x)
        x = x[:, 0, :]
        return x
