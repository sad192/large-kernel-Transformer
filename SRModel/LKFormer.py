# 测试
# 开发时间：2023/3/13 20:31
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import math
from models.PoolFormer import PoolFormerBlock,Pooling,PoolFormerT
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from models.basicblock import DWT,IWT
import numbers
from models.SwinT import SwinT
from models.bra import BiLevelRoutingAttention
from models.basicblock import CCALayer, PALayer
from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Upsample_SR(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample_SR, self).__init__(*m)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.SiLU):
        super().__init__()
        hidden_features = int(4*in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features,kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features,kernel_size=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn = ConvFFN_new(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
#########################################
### 近似21x21
class LKA_back(nn.Module):
    def __init__(self, dim):
        super(LKA_back,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.dwConv3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.dwConv5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.conv1_3 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)
        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        # self.conv1_3 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.dwConv3(attn)

        u1 = attn.clone()
        attn = self.conv1_2(attn)
        attn = self.dwConv5(attn)
        attn = self.act(attn+u1)

        u2 = attn.clone()
        attn = self.conv1_3(attn)
        attn = self.conv_spatial(attn)
        attn = self.act(attn + u2)

        u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        attn = self.act(attn + u3)

        attn = self.conv1_5(attn)
        return attn

class LKA_back_new(nn.Module):
    def __init__(self, dim):
        super(LKA_back_new,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.dwConv3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.dwConv5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.conv1_3 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)
        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.Conv21 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),nn.Conv2d(hidden, hidden, kernel_size=(1, 21), padding=(0, int(21 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(21, 1), padding=(int(21 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))
        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        # self.conv1_3 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.dwConv3(attn)

        u1 = attn.clone()
        attn = self.conv1_2(attn)
        attn = self.dwConv5(attn)
        attn = self.act(attn+u1)

        u2 = attn.clone()
        attn = self.conv1_3(attn)
        attn = self.conv_spatial(attn)
        attn = self.act(attn + u2)

        u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        attn = self.act(attn + u3)

        u4 = attn.clone()
        attn = self.Conv21(attn)
        attn = self.act(attn + u4)

        attn = self.conv1_5(attn)
        return attn

class LKA_back_new_attn(nn.Module):
    def __init__(self, dim):
        super(LKA_back_new_attn,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_0 = nn.Conv2d(dim, dim, kernel_size=1)

        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()

        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)

        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.Conv21 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),nn.Conv2d(hidden, hidden, kernel_size=(1, 21), padding=(0, int(21 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(21, 1), padding=(int(21 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.Conv31 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),
                                    nn.Conv2d(hidden, hidden, kernel_size=(1, 31), padding=(0, int(31 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(31, 1), padding=(int(31 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = self.conv1_0(x)

        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.conv_spatial(attn)


        u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        attn = self.act(attn + u3)

        u4 = attn.clone()
        attn = self.Conv21(attn)
        attn = self.act(attn + u4)

        u5 = attn.clone()
        attn = self.Conv31(attn)
        attn = self.act(attn + u5)

        # u6 = attn.clone()
        # attn = self.Conv41(attn)
        # attn = self.act(attn + u6)

        attn = self.conv1_5(attn)

        out1 = u * attn
        out1 = self.proj_1(out1)
        return out1

class LKA_back_new_attn_copy(nn.Module):
    def __init__(self, dim):
        super(LKA_back_new_attn_copy,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_0 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.dwConv3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.dwConv5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.conv1_3 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)
        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.Conv21 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),nn.Conv2d(hidden, hidden, kernel_size=(1, 21), padding=(0, int(21 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(21, 1), padding=(int(21 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.Conv31 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),
                                    nn.Conv2d(hidden, hidden, kernel_size=(1, 31), padding=(0, int(31 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(31, 1), padding=(int(31 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = self.conv1_0(x)

        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.dwConv3(attn)

        u1 = attn.clone()
        attn = self.conv1_2(attn)
        attn = self.dwConv5(attn)
        attn = self.act(attn+u1)

        u2 = attn.clone()
        attn = self.conv1_3(attn)
        attn = self.conv_spatial(attn)
        attn = self.act(attn + u2)

        u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        attn = self.act(attn + u3)

        u4 = attn.clone()
        attn = self.Conv21(attn)
        attn = self.act(attn + u4)

        u5 = attn.clone()
        attn = self.Conv31(attn)
        attn = self.act(attn + u5)

        attn = self.conv1_5(attn)

        out1 = u * attn
        out1 = self.proj_1(out1)
        return out1

class LKA_back_new_attn_no(nn.Module):
    def __init__(self, dim):
        super(LKA_back_new_attn_no,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_0 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.dwConv3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.dwConv5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.conv1_3 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)
        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.Conv21 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),nn.Conv2d(hidden, hidden, kernel_size=(1, 21), padding=(0, int(21 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(21, 1), padding=(int(21 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))
        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = self.conv1_0(x)

        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.dwConv3(attn)

        # u1 = attn.clone()
        attn = self.conv1_2(attn)
        attn = self.dwConv5(attn)
        # attn = self.act(attn+u1)

        # u2 = attn.clone()
        attn = self.conv1_3(attn)
        attn = self.conv_spatial(attn)
        # attn = self.act(attn + u2)

        # u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        # attn = self.act(attn + u3)

        # u4 = attn.clone()
        attn = self.Conv21(attn)
        # attn = self.act(attn + u4)

        attn = self.conv1_5(attn)

        out1 = u * attn
        out1 = self.proj_1(out1)
        return out1

## large kernel attention back 默认使用
class Attention_back(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(Attention_back, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # # self.qkv_dwconv = LKA_back(int(dim * 3)) ## ori 默认
        # self.qkv_dwconv = LKA_back_new(int(dim * 3)) ##new
        ############################ 分成三个特征矩阵
        self.q_dwconv = LKA_back_new(dim)
        self.k_dwconv = LKA_back_new(dim)
        self.v_dwconv = LKA_back_new(dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        ################### ori###################
        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)
        ################## 分成3个#################
        q = self.q_dwconv(x)
        k = self.k_dwconv(x)
        v = self.v_dwconv(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

## large kernel attention back
class Attention_back_three(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(Attention_back_three, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # # self.qkv_dwconv = LKA_back(int(dim * 3)) ## ori 默认
        # self.qkv_dwconv = LKA_back_new(int(dim * 3)) ##new
        ############################ 分成三个特征矩阵
        self.q_dwconv = LKA_back_new(dim)
        self.k_dwconv = LKA_back_new(dim)
        self.v_dwconv = LKA_back_new(dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        ################### ori###################
        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)
        ################## 分成3个#################
        q = self.q_dwconv(x)
        k = self.k_dwconv(x)
        v = self.v_dwconv(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)
        # print('qkv', attn.shape)
        out = (v @ attn)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

## inceptionAttn back
class MSPA_back_ori(nn.Module):
    def __init__(self, dim, head=4,distillation_rate=0.25, bias=False):
        super(MSPA_back_ori, self).__init__()
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)

        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv1_3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.Conv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1,
                               dilation=1, groups=dim)
        hiddenF = int(2*dim)

        self.conv1_0 = nn.Conv2d(dim, hiddenF, kernel_size=1, bias=bias)
        self.Conv21 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 21), padding=(0, int(21//2)), stride=1,
                                    dilation=1, groups=dim),
                                    nn.Conv2d(dim, dim, kernel_size=(21, 1), padding=(int(21//2), 0), stride=1,
                                    dilation=1, groups=dim))

        self.conv1_1 = nn.Conv2d(dim , hiddenF, kernel_size=1, bias=bias)
        self.Conv11 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=padding1, stride=1,
                                        dilation=1, groups=dim),
                                       nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=padding2, stride=1,
                                                 dilation=1, groups=dim))

        self.conv1_2 = nn.Conv2d(dim , hiddenF, kernel_size=1, bias=bias)
        self.Conv7 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0,3), stride=1,
                                        dilation=1, groups=dim),
                                     nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), stride=1,
                                               dilation=1, groups=dim))


        self.conv1_out = nn.Conv2d(int(3*hiddenF + dim), dim, kernel_size=1, bias=bias)
        self.act = nn.SiLU(inplace=True)
        # self.outConv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        u = x.clone()

       ## 先进行5x5的卷积特征提取
        x = self.conv1_3(x)
        x = self.Conv5(x)

        ## 计算特征图
        c2 = self.Conv7(x)
        c2 = self.conv1_2(c2)


        c3 = self.Conv11(x)
        c3 = self.conv1_1(c3)

        c4 = self.Conv21(x)
        c4 = self.conv1_0(c4)

        attn1 = torch.cat([x, c2, c3, c4], 1)
        attn1 = self.conv1_out(attn1)
        attn1 = self.act(attn1)

        out1 = attn1*u
        # out1 = self.outConv1(out1)
        return out1


class MSPA_new_back(nn.Module):
    def __init__(self, dim, head=4,distillation_rate=0.25, bias=False):
        super(MSPA_new_back, self).__init__()
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        hiddenF = int(dim) # 可以尝试扩大通道数

        self.conv1_3 = nn.Conv2d(dim, hiddenF, kernel_size=1, bias=bias)
        self.Conv3 = nn.Conv2d(hiddenF, hiddenF, kernel_size=3, padding=1, stride=1,
                                 dilation=1, groups=hiddenF)

        # self.act = nn.SiLU(inplace=True)

        # self.conv1_out = nn.Conv2d(int(5*hiddenF), dim, kernel_size=1, bias=bias)
        self.conv1_out = nn.Conv2d(int(1*hiddenF), dim, kernel_size=1, bias=bias)
        # self.CA = CCALayer(dim) int(4*hiddenF)

    def forward(self, x):
        # x1 = x.clone()
        x1 = self.conv1(x)

        c1 = self.conv1_3(x)
        c1 = self.Conv3(c1)
        # attn1 = torch.cat([ c2, c3], 1) ## c1, c2, c3, c4
        attn1 = self.conv1_out(c1) # attn1


        out1 = attn1*x1

        return out1

class MSPA_back(nn.Module):
    def __init__(self, dim, head=4,distillation_rate=0.25, bias=False):
        super(MSPA_back, self).__init__()
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.conv1_3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.Conv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1,
                               dilation=1, groups=dim)
        hiddenF = int(dim)

        self.conv1_0 = nn.Conv2d(dim, hiddenF, kernel_size=1, bias=bias)
        self.Conv21 = nn.Sequential(nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 21), padding=(0, int(21//2)), stride=1,
                                    dilation=1, groups=hiddenF),
                                    nn.Conv2d(hiddenF, hiddenF, kernel_size=(21, 1), padding=(int(21//2), 0), stride=1,
                                    dilation=1, groups=hiddenF))

        self.conv1_1 = nn.Conv2d(dim , hiddenF, kernel_size=1, bias=bias)
        self.Conv11 = nn.Sequential(nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 11), padding=padding1, stride=1,
                                        dilation=1, groups=hiddenF),
                                       nn.Conv2d(hiddenF, hiddenF, kernel_size=(11, 1), padding=padding2, stride=1,
                                                 dilation=1, groups=hiddenF))

        self.conv1_2 = nn.Conv2d(dim , hiddenF, kernel_size=1, bias=bias)
        self.Conv7 = nn.Sequential(nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 7), padding=(0,3), stride=1,
                                        dilation=1, groups=hiddenF),
                                     nn.Conv2d(hiddenF, hiddenF, kernel_size=(7, 1), padding=(3, 0), stride=1,
                                               dilation=1, groups=hiddenF))


        self.conv1_out = nn.Conv2d(int(3*hiddenF + dim), dim, kernel_size=1, bias=bias)
        # self.act = nn.SiLU(inplace=True)
        # self.outConv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        # u = x.clone()
        x1 = self.conv1(x)
       ## 先进行5x5的卷积特征提取
        x2 = self.conv1_3(x)
        x2 = self.Conv5(x2)

        ## 计算特征图
        c2 = self.conv1_2(x2)
        c2 = self.Conv7(c2)

        c3 = self.conv1_1(x2)
        c3 = self.Conv11(c3)

        c4 = self.conv1_0(x2)
        c4 = self.Conv21(c4)


        attn1 = torch.cat([x2, c2, c3, c4], 1)
        attn1 = self.conv1_out(attn1)

        out1 = attn1*x1
        # out1 = self.outConv1(out1)
        return out1

class inceptionAttn_back(nn.Module):
    def __init__(self,dim=64):
        super(inceptionAttn_back, self).__init__()
        hiddenC = int(2*dim)
        self.conv1_1 = nn.Conv2d(dim, hiddenC, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        # self.attn = MSPA_back(hiddenC) ##默认
        self.attn = MSPA_new_back(hiddenC) ### used
        self.conv1_2 = nn.Conv2d(hiddenC, dim, kernel_size=1, bias=True)
    def forward(self,x):
        # shorcut = x.clone()
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.attn(x)
        x = self.conv1_2(x)
        return x

class CAB_back(nn.Module):
    def __init__(self,dim=64):
        super(CAB_back, self).__init__()

        self.conv3_1 = nn.Conv2d(dim, dim, 3,1,1)
        self.act = nn.SiLU()
        self.conv3_2 = nn.Conv2d(dim, dim, 3,1,1)
        self.CA = CCALayer(dim)
    def forward(self,x):
        # shorcut = x.clone()
        x = self.conv3_1(x)
        x = self.act(x)
        x = self.conv3_2(x)
        x = self.CA(x)
        return x

class CAB_back_DW(nn.Module):
    def __init__(self,dim=64):
        super(CAB_back_DW, self).__init__()
        self.BSconv1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=False))

        self.act = nn.SiLU()
        self.BSconv2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),
                                     nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=False))

        self.CA = CCALayer(dim)
    def forward(self,x):
        # shorcut = x.clone()
        x = self.BSconv1(x)
        x = self.act(x)
        x = self.BSconv2(x)
        x = self.CA(x)
        return x

##### ParaHybrid block
class ParaHybridBlock_back(nn.Module):
    def __init__(self, dim, head=4,ffn_expansion_factor=4, distillation_rate=0.25, LayerNorm_type='WithBias',bias=False):
        super(ParaHybridBlock_back, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        # self.attn = Attention_back(dim,num_heads=head)
        self.attn = LKA_back_new_attn(dim)  #LKA_back_new_attn inceptionAttn_back
        # self.CA = CAB_back(dim)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = inceptionAttn_back(dim) # inceptionAttn_back
        # self.ffn = LKA_back_new_attn_no(dim)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.attn(x1) # + self.CA(x1)

        x = x + self.ffn(self.norm2(x))

        return x

#### stage block back
class ParaHybridStage_back(nn.Module):
    def __init__(self, dim, depth=4):
        super(ParaHybridStage_back, self).__init__()

        # self.basicBlock = nn.Sequential(*[
        #     ParaHybridBlock(dim=dim,head=heads[i]) for i in range(depth)])
        self.B1 = ParaHybridBlock_back(dim=dim)

        self.B2 = ParaHybridBlock_back(dim=dim)

        self.B3 = ParaHybridBlock_back(dim=dim)

        self.B4 = ParaHybridBlock_back(dim=dim)

        self.B5 = ParaHybridBlock_back(dim=dim)

        self.B6 = ParaHybridBlock_back(dim=dim)
        #
        # self.B7 = ParaHybridBlock_back(dim=dim)
        #
        # self.B8 = ParaHybridBlock_back(dim=dim)
        ### new
        # self.Conv1_out = nn.Conv2d(int(6*dim), dim, kernel_size=1)

        self.lastConv = nn.Conv2d(dim,dim,3,1,1)
        ###new
        # self.inPA = PALayer(dim)

    def forward(self, x):

        x1 = self.B1(x)

        # tem1 = self.Conv1_1(tem1)
        x2 = self.B2(x1)

        x3 = self.B3(x2)
        #
        x4 = self.B4(x3)

        x5 = self.B5(x4)

        x6 = self.B6(x5)
        out1 = self.lastConv(x6)+x
        return out1

#### pool
class PPM(nn.Module):
    def __init__(self, in_dim,  bins=(1, 2, 3, 6),LayerNorm_type='WithBias'):
        super(PPM, self).__init__()
        reduction_dim = int(in_dim / len(bins))
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                LayerNorm(reduction_dim, LayerNorm_type),
                nn.SiLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

##### ConvBlock ori
class ConvBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias', bias=True):
        super(ConvBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=2 * dim)  # depthwise conv
        self.act = nn.SiLU(inplace=True)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.CA = CCALayer(dim * 4)
        self.tras_conv1 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.conv1(out1)
        out1 = self.act(out1)
        out1 = self.dwconv(out1)
        out1 = self.CA(out1)
        out1 = self.tras_conv1(out1)
        out1 = self.act(out1 + x)
        return out1

class LargeKernelB(nn.Module):
    def __init__(self,dim=64):
        super(LargeKernelB, self).__init__()
        hiddenC = int(4*dim)
        self.conv1_1 = nn.Conv2d(dim, hiddenC, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(hiddenC, hiddenC, 5, padding=2, groups=int(hiddenC), bias=True)
        self.conv1_2 = nn.Conv2d(hiddenC, hiddenC, kernel_size=1, bias=True)
        self.conv_spatial = nn.Conv2d(hiddenC, hiddenC, 7, stride=1, padding=9, groups=int(hiddenC), dilation=3, bias=True)
        self.conv1_3 = nn.Conv2d(hiddenC, dim, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self,x):
        x1 = self.conv1_1(x)
        x1 = self.conv5(x1)
        x1 = self.conv1_2(x1)
        x1 = self.conv_spatial(x1)
        x1 = self.conv1_3(x1)
        out = self.act(x1 + x)
        return out

class ConvFFN_new(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias', bias=True):
        super(ConvFFN_new, self).__init__()
        self.conv1_1 = nn.Conv2d(dim , dim * 4, kernel_size=1, bias=bias)
        self.dwconv3_1 = nn.Conv2d(dim*4 , dim * 4, kernel_size=3, padding=1, groups=int(dim *4),bias=bias)  # depthwise conv
        self.act = nn.SiLU(inplace=True)
        self.conv1_2 = nn.Conv2d(dim * 4 , dim * 4, kernel_size=1, bias=bias)
        self.dwconv3_2 = nn.Conv2d(dim * 4 , dim * 4, kernel_size=3, padding=1, groups=int(dim * 4), bias=bias)  # depthwise conv
        self.tras_conv1 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.dwconv3_1(out1)
        out1 = self.conv1_2(self.act(out1))
        out1 = self.dwconv3_2(out1) ## + out1
        out1 = self.tras_conv1(out1)
        return out1

class ConvFFN_1(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias', bias=False):
        super(ConvFFN_1, self).__init__()
        self.conv1_1 = nn.Conv2d(dim , dim * 4, kernel_size=1, bias=bias)
        self.dwconv3_1 = nn.Conv2d(dim*4 , dim * 4, kernel_size=3, padding=1, groups=int(dim *4),bias=bias)  # depthwise conv
        self.act = nn.SiLU(inplace=True)
        self.tras_conv1 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.dwconv3_1(out1)
        out1 = self.act(out1)
        out1 = self.tras_conv1(out1)
        return out1

## inceptionAttn
class MSPA(nn.Module):
    def __init__(self, dim, head=4,distillation_rate=0.25, bias=False):
        super(MSPA, self).__init__()
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        hiddenF = int(dim)

        self.conv1_0 = nn.Conv2d(dim, hiddenF, kernel_size=1, bias=bias)
        self.Conv21 = nn.Sequential(nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 21), padding=(0, int(21//2)), stride=1,
                                    dilation=1, groups=hiddenF),
                                    nn.Conv2d(hiddenF, hiddenF, kernel_size=(21, 1), padding=(int(21//2), 0), stride=1,
                                    dilation=1, groups=hiddenF))

        self.conv1_1 = nn.Conv2d(dim , hiddenF, kernel_size=1, bias=bias)
        self.Conv11 = nn.Sequential(nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 11), padding=padding1, stride=1,
                                        dilation=1, groups=hiddenF),
                                       nn.Conv2d(hiddenF, hiddenF, kernel_size=(11, 1), padding=padding2, stride=1,
                                                 dilation=1, groups=hiddenF))

        self.conv1_2 = nn.Conv2d(dim , hiddenF, kernel_size=1, bias=bias)
        self.Conv7 = nn.Sequential(nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 7), padding=(0,3), stride=1,
                                        dilation=1, groups=hiddenF),
                                     nn.Conv2d(hiddenF, hiddenF, kernel_size=(7, 1), padding=(3, 0), stride=1,
                                               dilation=1, groups=hiddenF))

        self.conv1_3 = nn.Conv2d(dim, hiddenF, kernel_size=1, bias=bias)
        self.Conv5 = nn.Conv2d(hiddenF, hiddenF, kernel_size=5, padding=2, stride=1,
                                 dilation=1, groups=hiddenF)

        self.act = nn.SiLU(inplace=True)

        self.conv1_out = nn.Conv2d(int(4*hiddenF), dim, kernel_size=1, bias=bias)


    def forward(self, x):

        x1 = self.conv1(x)

        c1 = self.conv1_3(x)
        c1 = self.Conv5(c1)

        c2 = self.conv1_2(x)
        c2 = self.Conv7(c2)

        c3 = self.conv1_1(x)
        c3 = self.Conv11(c3)

        c4 = self.conv1_0(x)
        c4 = self.Conv21(c4)

        attn1 = torch.cat([c1, c2, c3, c4], 1)
        attn1 = self.conv1_out(attn1)

        out1 = attn1*x1

        return out1

class inceptionAttn(nn.Module):
    def __init__(self,dim=64):
        super(inceptionAttn, self).__init__()
        hiddenC = int(dim)
        self.conv1_1 = nn.Conv2d(dim, hiddenC, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.attn = MSPA(hiddenC)
        self.conv1_2 = nn.Conv2d(hiddenC, dim, kernel_size=1, bias=True)
    def forward(self,x):
        # shorcut = x.clone()
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.attn(x)
        x = self.conv1_2(x)
        return x

## 考虑不使用信息蒸馏技术
class attn_new(nn.Module):
    def __init__(self, dim, head=4,distillation_rate=0.25, bias=False):
        super(attn_new, self).__init__()
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.hiddenC = int(dim * distillation_rate)  ## 分成四份

        self.attn1 = Attention(self.hiddenC,num_heads=head)

        hiddenF = int(self.hiddenC*4)
        self.conv1_1 = nn.Conv2d(self.hiddenC , hiddenF, kernel_size=1, bias=bias)
        self.Lconv1 = nn.Conv2d(hiddenF, hiddenF, kernel_size=(1, 11), padding=padding1, stride=1,
                                dilation=1, groups=hiddenF)
        self.Lconv2 = nn.Conv2d(hiddenF, hiddenF, kernel_size=(11, 1), padding=padding2, stride=1,
                                dilation=1, groups=hiddenF)
        self.conv1_2 = nn.Conv2d(hiddenF , hiddenF, kernel_size=1, bias=bias)
        self.Conv7 = nn.Conv2d(hiddenF, hiddenF, kernel_size=(7, 7), padding=3, stride=1,
                                dilation=1, groups=hiddenF)
        self.conv1_3 = nn.Conv2d(hiddenF, self.hiddenC, kernel_size=1, bias=bias)

        self.pool = PPM(self.hiddenC)

        self.conv1_4 = nn.Conv2d(self.hiddenC, self.hiddenC, kernel_size=1, bias=bias)

        self.act = nn.SiLU(inplace=True)

        self.conv1_out = nn.Conv2d(int(dim + self.hiddenC), dim, kernel_size=1, bias=bias)


    def forward(self, x):
        distilled_c1, distilled_c2,pool_c1,indenty_c1 = torch.split(x, (self.hiddenC, self.hiddenC,self.hiddenC,self.hiddenC), dim=1)

        distilled_c1 = self.attn1(distilled_c1)

        distilled_c2 = self.conv1_1(distilled_c2)
        distilled_c2 = self.Lconv1(distilled_c2)
        distilled_c2 = self.Lconv2(distilled_c2)
        distilled_c2 = self.act(distilled_c2)
        distilled_c2 = self.conv1_2(distilled_c2)
        distilled_c2 = self.Conv7(distilled_c2)
        distilled_c2 = self.act(distilled_c2)
        distilled_c2 = self.conv1_3(distilled_c2)

        pool_c1 = self.pool(pool_c1)

        indenty_c1 = self.conv1_4(indenty_c1)

        out1 = torch.cat([distilled_c1, distilled_c2, pool_c1, indenty_c1], 1)
        out1 = self.conv1_out(out1)

        return out1

##### ParaHybrid block
#### 该模块也有效 32.75dB 1000epochs
class ParaHybridBlock(nn.Module):
    def __init__(self, dim, head=4,ffn_expansion_factor=4, distillation_rate=0.25, LayerNorm_type='WithBias',bias=False):
        super(ParaHybridBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        # self.attn = inceptionAttn(dim)
        # self.attn = inceptionAttn_back(dim)
        self.attn = LKA_back_new_attn(dim)

        # self.largeKernelB = LargeKernelB(dim)
        # self.transformerB = TransformerBlock(dim,num_heads=head)
        # # self.ConvTransformerT = ConvformerBlock(dim)
        #
        # self.conv1_1 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = ConvFFN_1(dim)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = ConvFFN_new(dim)
        # self.ffn = ConvFFN_new(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

#### stage block
class ParaHybridStage(nn.Module):
    def __init__(self, dim, depth=4):
        super(ParaHybridStage, self).__init__()

        # self.basicBlock = nn.Sequential(*[
        #     ParaHybridBlock(dim=dim,head=heads[i]) for i in range(depth)])
        self.B1 = ParaHybridBlock(dim=dim,head=1)
        # self.Conv1_1 = nn.Conv2d(2*dim, dim, kernel_size=1)
        self.B2 = ParaHybridBlock(dim=dim,head=2)
        # self.Conv1_2 = nn.Conv2d(3*dim, dim, kernel_size=1)
        self.B3 = ParaHybridBlock(dim=dim,head=4)
        # self.Conv1_3 = nn.Conv2d(4*dim, dim, kernel_size=1)
        self.B4 = ParaHybridBlock(dim=dim,head=8)
        # self.Conv1_4 = nn.Conv2d(5 * dim, dim, kernel_size=1)
        self.B5 = ParaHybridBlock(dim=dim, head=8)
        # self.Conv1_5 = nn.Conv2d(6 * dim, dim, kernel_size=1)
        self.B6 = ParaHybridBlock(dim=dim, head=8)
        # self.act = nn.SiLU(inplace=True)
        self.lastConv = nn.Conv2d(dim,dim,3,1,1)

    def forward(self, x):
        # x1 = self.basicBlock(x)
        # x1 = self.Conv1_3(x1)
        x1 = self.B1(x)
        # x1 = self.act(x1)
        # tem1 = torch.cat([x1, x], 1)
        # tem1 = self.Conv1_1(tem1)
        x2 = self.B2(x1)
        # x2 = self.act(x2)
        # tem2 = torch.cat([x2, x1,x], 1)
        # tem2 = self.Conv1_2(tem2)
        x3 = self.B3(x2)
        # x3 = self.act(x3)
        # tem3 = torch.cat([x3,x2, x1,x], 1)
        # tem3 = self.Conv1_3(tem3)
        x4 = self.B4(x3)
        # x4 = self.act(x4)
        # tem4 = torch.cat([x4,x3,x2, x1,x], 1)
        # tem4 = self.Conv1_4(tem4)
        x5 = self.B5(x4)
        # x5 = self.act(x5)
        # tem5 = torch.cat([x5,x4, x3, x2, x1, x], 1)
        # tem5 = self.Conv1_5(tem5)
        x6 = self.B6(x5)
        out1 = self.lastConv(x6)+x
        return out1

class unetB(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 img_range=1.,
                 upscale=4,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(unetB, self).__init__()

        self.encoder_level1 = nn.Sequential(*[ParaHybridBlock(dim,heads[i]) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[1])])

        # self.fuse_multiple_l1andl2 = fuse_multiple_scale_map(dim,int(dim * 2 ** 1))
        #
        # self.fuse_multiple_l2andl3 = fuse_multiple_scale_map(int(dim * 2 ** 1),int(dim * 2 ** 2))

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 3),heads[i]) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[0])])


    def forward(self, x):

        ### unet先处理LR上采样两倍的图片
        out_enc_level1 = self.encoder_level1(x)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        #
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        ################### ori
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        ############## ori
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        ############################
        inp_dec_level1 = self.up2_1(out_dec_level2)
        #### ori
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        #################
        out_dec_level1 = self.refinement(out_dec_level1)

        return out_dec_level1


##########################################################################
##---------- Restormer -----------------------
###### dim可以调整为64
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 img_range=1.,
                 upscale=4,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()
        self.img_range = img_range
        self.upscale = upscale
        num_feat = 64
        if inp_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        ########## 使用卷积对LR图片的尺寸进行放大,将两步整合为一步#############
        self.LR_1_conv3 = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        ######### x2 2h,2w,c
        # self.LR_up_x2 = nn.Sequential(nn.Conv2d(dim, 4 * dim, 3, 1, 1),
        #                                   nn.PixelShuffle(2))
        ######## x4 4h,4w,c
        # self.LR_up_x4 = nn.Sequential(nn.Conv2d(dim, 4 * dim, 3, 1, 1),
        #                                   nn.PixelShuffle(2))

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[ParaHybridBlock(dim,heads[i]) for i in range(num_blocks[0])])

        # self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        # self.encoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[1])])

        self.down1_2 = DWT()
        self.encoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[1])])

        # self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        # self.encoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[2])])
        self.down2_3 = DWT()  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 4), heads[i]) for i in range(num_blocks[2])])

        # self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        # self.latent = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 3),heads[i]) for i in range(num_blocks[3])])
        self.down3_4 = DWT()  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 6), heads[i]) for i in range(num_blocks[3])])

        # self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2),heads[i]) for i in range(num_blocks[2])])
        self.up4_3 = IWT()  ## From Level 4 to Level 3
        self.decoder_level3 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 4), heads[i]) for i in range(num_blocks[2])])

        # self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[1])])
        self.up3_2 = IWT()  ## From Level 3 to Level 2
        self.decoder_level2 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 2), heads[i]) for i in range(num_blocks[1])])

        # self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        # self.decoder_level1 = nn.Sequential(*[ParaHybridBlock(int(dim * 2 ** 1),heads[i]) for i in range(num_blocks[0])])
        self.up2_1 = IWT()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[ParaHybridBlock(int(dim), heads[i]) for i in range(num_blocks[0])])

        #### 对LR输入的处理 ###########################
        self.LR_act = torch.nn.SiLU(inplace=True)  # SiLU

        self.refinement = nn.Sequential(*[ParaHybridBlock(int(dim),heads[i]) for i in range(num_refinement_blocks)])

        # for classical SR
        # self.conv_before_upsample = nn.Sequential(nn.Conv2d(int(dim), num_feat, 3, 1, 1),
        #                                           nn.SiLU(inplace=True))
        # self.upsample = Upsample_SR(upscale, num_feat)
        # self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1)
        ##########################################################

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        #############################################
        ######### 上采样后的LR特征提取以及重建过程
        H, W = inp_img.shape[2:]
        ########## 归一化
        self.mean = self.mean.type_as(inp_img)
        inp_img = (inp_img - self.mean) * self.img_range
        ######## 插值上采样
        x_up = F.interpolate(inp_img, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        #### 利用卷积上采样
        x = self.LR_1_conv3(x_up)

        ### unet直接处理LR图片
        # inp_enc_level1 = self.patch_embed(inp_img)
        ### unet先处理LR上采样两倍的图片
        out_enc_level1 = self.encoder_level1(x)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        #
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        ### 小波变换
        inp_dec_level3 = self.up4_3(latent) + out_enc_level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3)



        ############## ori
        ### 小波变换
        inp_dec_level2 = self.up3_2(out_dec_level3) + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        ##### 解码层，融合编码层对应的多尺度特征 l1 and l2
        # fuse_l1andl2 = self.fuse_multiple_l1andl2(out_enc_level1,out_enc_level2)

        ############################

        #### ori
        ### 小波变换
        inp_dec_level1 = self.up2_1(out_dec_level2) + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        ######## 在最后进行上采样 x2 -> refine -> x2 -> refine
        # out_dec_level1 = self.upsample_1(out_dec_level1)
        #################
        out_dec_level1 = self.refinement(out_dec_level1)

        ##### 在最后再上采样
        out_dec_level1 = self.output(out_dec_level1)  #+ x_up

        out_dec_level1 = out_dec_level1 / self.img_range + self.mean
        # x[:, :, :H * self.upscale, :W * self.upscale]
        return out_dec_level1[:, :, :H * self.upscale, :W * self.upscale]

if __name__ == '__main__':

    input = torch.rand(2, 64, 64, 64)
    # model = Restormer(dim=64)
    model = Attention_back_three(64)
    out = model(input)
    print('out:', out.shape)