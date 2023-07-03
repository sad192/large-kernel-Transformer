# 测试
# 开发时间：2023/1/15 14:21
import math
# from utils import module_util as mutil
import torch.nn.functional as F
import torch
from einops import rearrange
from thop import profile
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
# from .FCVit import BasicBlock as FCVitBlock
from models.PoolFormer import PoolFormerBlock,Pooling,PoolFormerT
from models.basicblock import  BSConvURB, PALayer,CCALayer,LKAAttention,FeedForward,ConvFFN,spacialBias,LargeKernelB,IWT,DWT
from swinUnetSR.restormer import LayerNorm
from swinUnetSR.restormer_swin_light import ParaHybridBlock

from models.hornet import Block as gnConvBlock
# from models.deformerConv import DeformConv2d
# from models.CCA import CrissCrossAttention
# from models.Coordatt import CoordAtt
# from models.ASPP import LCBP
# from timm.models.layers import  get_padding
# from models.RFConv import RFConv2d
params= {
    "global_context":{
        "weighted_gc": True,
        "gc_reduction": 8,
        "compete": False,
        "head": 8,
    },
    "spatial_mixer":{
        "use_globalcontext":True,
        "useSecondTokenMix": True,
        "mix_size_1": 5,
        "mix_size_2": 7,
        "fc_factor": 8,
        "fc_min_value": 16,
        "useSpatialAtt": True
    },
    "channel_mixer":{
        "useChannelAtt": True,
        "useDWconv":True,
        "DWconv_size":3
    },
    "spatial_att":{
        "kernel_size": 3,
        "dim_reduction":8
    },
    "channel_att":{
        "size_1": 3,
        "size_2": 5,
    }
}

default_search_cfg = dict(
    num_branches=3,
    expand_rate=0.5,
    max_dilation=None,
    min_dilation=1,
    init_weight=0.01,
    search_interval=1250,
    max_search_step=0,
)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }
default_cfgs = {
    's': _cfg(crop_pct=0.9),
    'm': _cfg(crop_pct=0.95),
}
fcvt_params = params.copy()

fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
fcvt_params["spatial_mixer"]["use_globalcontext"]=True
fcvt_params["spatial_mixer"]["mix_size_1"] = 11
fcvt_params["spatial_mixer"]["mix_size_2"]=11

fcvt_params["global_context"]["weighted_gc"] = True
fcvt_params["global_context"]["head"] = 8
fcvt_params["global_context"]["compete"] = True

fcvt_params["channel_mixer"]["useDWconv"] = True

fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
fcvt_params["channel_mixer"]["useChannelAtt"] = False

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Upsample(nn.Sequential):
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
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class LCB(nn.Module):
    def __init__(self,dim=48, dilation=1,
                 norm=GroupNorm,
                 block_lk_size=1):
        super().__init__()

        self.norm1 = GroupNorm(dim)
        padding = block_lk_size // 2
        self.depthwise1 = nn.Conv2d(dim, dim, kernel_size=5,stride=1, padding=2, dilation=1, groups=dim)
        # self.CCA1 = CCALayer(dim)
        self.norm2 = GroupNorm(dim)
        self.depthwise2 = nn.Conv2d(dim, dim, kernel_size=block_lk_size, stride=1, padding=padding, dilation=1, groups=dim)
        self.norm3 = GroupNorm(dim)

        # self.pointwise2 = nn.Conv2d(dim, dim, 1)
        # self.CCA2 = CCALayer(dim)

        # self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        # self.depthwise_dilated = RFConv2d(
        #     in_channels=dim,
        #     out_channels=dim,
        #     kernel_size=7,
        #     stride=stride,
        #     padding=get_padding(kernel_size=7, stride=stride, dilation=dilation),
        #     dilation=dilation,
        #     groups=dim,
        #     bias=conv_bias,
        #     **search_cfgs)

    def forward(self, x):
        # u = x.clone()
        x1 = self.depthwise1(x) # self.pointwise1(x) self.CCA1(x)

        x2 = self.depthwise2(x)

        # x = self.pointwise2(x) # 测试 self.pointwise2(x) self.CCA2(x)
        x = self.norm1(x)
        x1 = self.norm2(x1)
        x2 = self.norm3(x2)
        # x = self.depthwise_dilated(x)
        return x + x1 + x2


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class ESA(nn.Module):
    def __init__(self, num_feat=48, conv=BSConvU, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

# inplans, planes: 输入通道，输出通道
# 不会改变输入的H和W
class PSAModule(nn.Module):
    # dim=96 conv_groups=[1, 4, 8, 12], dim=64 conv_groups=[1, 4, 8, 16]
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

# 串行提取
class buildBlock(nn.Module):
    def __init__(self, inc=48,outc=48,norm_layer=GroupNorm):
        super().__init__()
        # self.conv3 = nn.Conv2d(inc,outc,3, 1, 1, bias=True)
        self.conv1_1 = nn.Conv2d(int(inc),outc,kernel_size=1)
        # self.norm = norm_layer(inc)

        # FCVit自注意力
        # self.FcvAtten = FCVitBlock(dim=inc,mlp_ratio=8.,
        #          act_layer=nn.SiLU, norm_layer=GroupNorm,
        #          drop=.0, drop_path=0.,
        #          use_layer_scale=True, layer_scale_init_value=1e-5,params=fcvt_params) # 不改变 H W

        # PoolFormer自注意力
        self.PoolAtten = PoolFormerBlock(dim=inc,use_layer_scale=False) # 不改变 H W
        # self.PoolAtten = PoolFormerT(dim=inc) # 已测试
        # self.CCAatten = CrissCrossAttention(inc)

        # BSRB
        self.BSRB = BSConvURB(inc, outc, kernel_size=3)

        # self.maxPool = Pooling(pool_size=3) # 提取边缘特征 已测试
        # self.GAP =    # 使用全局平均池化层提取纹理的空间特征

        # 可变卷积
        # self.deformerConv = DeformConv2d(inc,outc) # 不改变 H W

        # 金字塔通道注意力
        # self.PsaAtten = PSAModule(inc,outc) # 不改变 H W
        # self.ESA = ESA(n_feats=inc, conv=nn.Conv2d)
        # self.CoordAtt = CoordAtt(inc,outc) # 位置通道注意力
        self.CCA = CCALayer(outc)
        # self.act = nn.SiLU(inplace=True)
        self.pixel_norm = nn.LayerNorm(outc)
        # mutil.initialize_weights([self.pixel_norm], 0.1)
    def forward(self, x):
        # 提取边缘特征
        # x2_1 = self.maxPool(x)
        # 考虑添加一个Conv1
        x0 = self.conv1_1(x) # self.CCA(x)

        # 提取全局特征信息
        x1 = self.PoolAtten(x0) # 已尝试使用MaxPool(有效)
        # x2 = self.deformerConv(x1)
        # x1 = self.CCAatten(x)

        # 提取局部特征信息
        # x1 = self.conv3(x)
        # x1 = self.act(x1)
        ##################
        # 替换为BSRB
        x2 = self.BSRB(x1)

        # x_3 = x2 + x           # torch.cat([x1, x2], dim=1)
        # x_3 = self.conv1_1(x_3)
        # x_3 = self.CCA(x_3)
                                    # x_3 = self.ESA(x_3) # 效果不好
        x_3 = self.CCA(x2)
        # x_3 = self.CoordAtt(x2)
        x_4 = x_3 + x

        # 添加 pixel_norm
        x_4 = x_4.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_4 = self.pixel_norm(x_4)
        x_4 = x_4.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return x_4

class dealChannelBlock(nn.Module):
    def __init__(self, dim=64):
        super(dealChannelBlock, self).__init__()

        self.conv1_1 = nn.Conv2d(dim,dim,kernel_size=1) #2*dim
        self.activation = torch.nn.SiLU(inplace=True)  #####nn.GELU()
        self.conv3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.CA = CCALayer(dim)
        # self.CA = LCA(2*dim)
        # self.CA = eca_layer(dim)
    def forward(self, x):
        ## 新增了一个跳跃连接
        shorcut = x.clone()
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.conv3_1(x)
        x = self.CA(x)
        x = x + shorcut
        return x

class CSBlock(nn.Module):
    def __init__(self, embed_dim=64, nf=64,distillation_rate=0.50):
        super(CSBlock, self).__init__()
        #### 新加的
        # self.pool = Pooling(int(embed_dim * 0.25))
        self.hiddenC = int(embed_dim * 0.25)
        self.identy = int(embed_dim * 0.5)
        # 知识蒸馏
        # self.distilled_channels = int(embed_dim * distillation_rate)
        # self.remaining_channels = int(embed_dim - self.distilled_channels)
        # self.distillation_rate = distillation_rate

        self.dealC = dealChannelBlock(self.hiddenC) ##self.remaining_channels
        self.dealS = LKAAttention(self.hiddenC) ##self.distilled_channels

        self.conv1_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        # self.dwConv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3,padding=1,groups=embed_dim)
        #### 最后可以考虑加一个PA层

    def forward(self, x):
        ### 新加的
        # x_pool = self.pool(x)
        # distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)
        channel_c1, pixel_c1,indenty_c1 = torch.split(x, (self.hiddenC, self.hiddenC,self.identy), dim=1)
        c1 = self.dealC(channel_c1)
        s1 = self.dealS(pixel_c1)

        # p1 = self.pool(pool_c1)

        out1 = torch.cat([c1, s1, indenty_c1], 1)
        ## new
        # out1 = x_pool + out1

        out1 = self.conv1_out(out1)
        # ##### new
        # out1 = self.dwConv3(out1)

        ###### 可以考虑就加一层 Conv1
        ####### rearrange(x, 'b (g d) h w -> b (d g) h w', g=8)
        return rearrange(out1, 'b (g d) h w -> b (d g) h w', g=8)

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

# 串行提取,关注通道+空间
class buildAllCNNBlock(nn.Module):
    def __init__(self, dim,num_block=2,LayerNorm_type='WithBias',bias=False):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.PoolAtten = PoolFormerBlock(dim=dim, use_layer_scale=False)
        # self.attn = gnConvBlock(dim)
        self.attn = nn.Sequential(*[CSBlock(dim) for i in range(num_block)]) #CSBlock(dim)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = ConvFFN(dim,expand=2)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# 大核残差卷积块
class LargeKernelBlock(nn.Module):
    def __init__(self, embed_dim,distillation_rate=0.25,LayerNorm_type='WithBias',bias=False):
        super().__init__()
        # 知识蒸馏
        self.distilled_channels_1 = int(embed_dim * distillation_rate)
        self.remaining_channels_1 = int(embed_dim - self.distilled_channels_1)
        self.distilled_channels_2 = int(self.remaining_channels_1 * distillation_rate)
        self.remaining_channels_2 = int(self.remaining_channels_1 - self.distilled_channels_2)
        self.distilled_channels_3 = int(self.remaining_channels_2 * distillation_rate)
        self.remaining_channels_3 = int(self.remaining_channels_2 - self.distilled_channels_3)

        self.distillation_rate = distillation_rate

        # self.conv1_1 = nn.Conv2d(self.distilled_channels_1,self.distilled_channels_1,kernel_size=1)
        self.LKRB_1 = LargeKernelB(self.remaining_channels_1)
        # self.conv1_2 = nn.Conv2d(self.distilled_channels_2,self.distilled_channels_2,kernel_size=1)
        self.LKRB_2 = LargeKernelB(self.remaining_channels_2)
        # self.conv1_3 = nn.Conv2d(self.distilled_channels_3,self.distilled_channels_3,kernel_size=1)
        self.LKRB_3 = LargeKernelB(self.remaining_channels_3)

        self.conv1_out = nn.Conv2d(embed_dim,embed_dim,kernel_size=1)

    def forward(self, x):
        distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels_1, self.remaining_channels_1), dim=1)
        remaining_c1 = self.LKRB_1(remaining_c1)
        distilled_c2, remaining_c2 = torch.split(remaining_c1, (self.distilled_channels_2, self.remaining_channels_2), dim=1)
        remaining_c2 = self.LKRB_2(remaining_c2)
        distilled_c3, remaining_c3 = torch.split(remaining_c2, (self.distilled_channels_3, self.remaining_channels_3), dim=1)
        remaining_c3 = self.LKRB_3(remaining_c3)
        out1 = torch.cat([distilled_c1, distilled_c2, distilled_c3, remaining_c3], 1)
        out1 = self.conv1_out(out1)
        return rearrange(out1, 'b (g d) h w -> b (d g) h w', g=8)

# class IMDModule(nn.Module):
#     def __init__(self, in_channels, distillation_rate=0.25):
#         super(IMDModule, self).__init__()
#         self.distilled_channels = int(in_channels * distillation_rate)
#         self.remaining_channels = int(in_channels - self.distilled_channels)
#         self.c1 = conv_layer(in_channels, in_channels, 3)
#         self.c2 = LargeKernelB(self.remaining_channels, in_channels, 3)
#         self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
#         self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
#         self.act = activation('lrelu', neg_slope=0.05)
#         self.c5 = conv_layer(in_channels, in_channels, 1)
#         self.cca = CCALayer(self.distilled_channels * 4)
#
#     def forward(self, input):
#         out_c1 = self.act(self.c1(input))
#         distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
#         out_c2 = self.act(self.c2(remaining_c1))
#         distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
#         out_c3 = self.act(self.c3(remaining_c2))
#         distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
#         out_c4 = self.c4(remaining_c3)
#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
#         out_fused = self.c5(self.cca(out)) + input
#         return out_fused

class unetSR(nn.Module):
    def __init__(self,
                 inc=3,
                 dim=48,
                 upscale=4,
                 upsampler='pixelshuffledirect', img_range=1.,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(unetSR, self).__init__()

        self.upsampler = upsampler
        self.img_range = img_range
        self.upscale = upscale
        if inc == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        #####################################################################################################
        ################################### 1, shallow feature extraction ############
        self.conv_first = nn.Conv2d(inc, dim, 3, 1, 1)

        # self.trans1 = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1),nn.SiLU(),nn.Conv2d(dim,dim,kernel_size=3, padding=1, groups=dim))
        # self.trans2 = nn.Sequential(nn.Conv2d(int(dim*2**1),int(dim*2**1),kernel_size=1),nn.SiLU(),nn.Conv2d(int(dim*2**1),int(dim*2**1),kernel_size=3, padding=1, groups=int(dim*2**1)))

        self.encoder_level1 = buildAllCNNBlock(dim)

        self.down1_2 = DWT()  ## From Level 1 to Level 2
        self.encoder_level2 = buildAllCNNBlock(int(dim*2**2))

        self.down2_3 = DWT()  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[buildAllCNNBlock(int(dim*2**4)) for i in range(1)])

        self.up3_2 = IWT()  ## From Level 3 to Level 2
        self.decoder_level2 = buildAllCNNBlock(int(dim*2**2))

        self.up2_1 = IWT()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = buildAllCNNBlock(dim)

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(dim, 64, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, 64)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, dim, dim,
                                            (64, 64))
        self.refine = buildAllCNNBlock(dim)
        self.conv_last = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_up = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        x = self.conv_first(x)  # 经过浅层特征提取

        out_enc_level1 = self.encoder_level1(x)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)


        inp_dec_level2 = self.up3_2(out_enc_level3)
        ############## ori
        out_dec_level2 = self.decoder_level2(inp_dec_level2+out_enc_level2)

        ############################
        inp_dec_level1 = self.up2_1(out_dec_level2)
        #### ori
        out_dec_level1 = self.decoder_level1(inp_dec_level1+out_enc_level1)
        #################

        x = self.upsample(out_dec_level1)   # 图像上采样重建 + x_up

        x = self.refine(x)
        x = self.conv_last(x)+x_up

        x = x / self.img_range + self.mean
        # ILR = F.interpolate(u, scale_factor=self.scale, mode='bilinear', align_corners=False)
        # x[:, :, :H * self.upscale, :W * self.upscale] + ILR
        return x[:, :, :H * self.upscale, :W * self.upscale]



######## 之前为64
class myNet(nn.Module):
    def __init__(self, inc=3,outc=3,upscale=4,embed_dim=48,depth=4,heads=[1, 2, 4, 8,16,24], img_size=64,upsampler='pixelshuffledirect', img_range=1.,norm_layer=GroupNorm):
        super(myNet, self).__init__()

        self.upsampler = upsampler
        self.img_range = img_range
        self.upscale = upscale
        if inc == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        #####################################################################################################
        ################################### 1, shallow feature extraction ############
        self.conv_first = nn.Conv2d(inc, embed_dim, 3, 1, 1)
        # self.conv_first = BSConvU(inc, embed_dim)
        # self.dwt = DWT()
        # self.iwt = IWT()
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = depth
        self.layers = nn.ModuleList()  # 存放深度特征提取模块
        # self.SB1 = spacialBias(embed_dim)
        # self.SB2 = spacialBias(int(embed_dim+4))
        # self.conv1_SB = nn.Conv2d(int(embed_dim+8),embed_dim,kernel_size=1)
        for i_layer in range(self.num_layers):
            layer = ParaHybridBlock(embed_dim) # buildBlock buildAllCNNBlock(inc=embed_dim,outc=embed_dim,kernel_size=large_kernel_sizes[i_layer])
            self.layers.append(layer)

        # self.last_deep_Conv = nn.Sequential(nn.Conv2d(depth*embed_dim, embed_dim,kernel_size=1),nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
        self.last_deep_Conv = nn.Sequential(nn.Conv2d(depth*embed_dim, embed_dim, kernel_size=1),nn.SiLU(),nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim))
        #####################################################################################################
        ################################### 2.2, 深度特征融合模块 ######################################
        ####################self.conv1 = nn.Conv2d(depth * embed_dim, embed_dim, kernel_size=1)
        # self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # depth*embed_dim
        # self.conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)
        # # self.conv3 = BSConvU(embed_dim, embed_dim)
        # self.last_deep_Conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),nn.SiLU(),nn.Conv2d(embed_dim, embed_dim, kernel_size=1))
        self.PA = PALayer(embed_dim)
        # self.PPM = PPM(embed_dim)
        #####################      ################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, 64, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, 64)
            self.conv_last = nn.Conv2d(64, outc, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, int(1*embed_dim), outc,
                                            (img_size, img_size))

    def forward_features(self, x):  # 经过深层特征提取（HRBCT）之后的输出
        retainV = []

        for layer in self.layers:
            x = layer(x)
            retainV.append(x)  # 暂时取消特征聚合操作

        # for i in range(self.num_layers):
        #     if i>2:
        #         if i==3:
        #             x = self.conv1_SB(x)
        #         x = self.layers[i](x)
        #     else:
        #         if i==0:
        #             x = self.SB1(x)
        #         elif i==1:
        #             x = self.SB2(x)
        #         x = self.layers[i](x)
        #     retainV.append(x)

        # depth=4 改进2：将每个HRBCT模块的输出 concat到一起，然后输入到深度特征融合层
        x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3]), 1)
        # depth=6
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5]), 1).contiguous()
        # x = self.last_deep_Conv(x1)
        return x1

    def residual_forward(self,x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x+x1)
        x3 = self.layers[2](x+x1+x2)
        x4 = self.layers[3](x+x1+x2+x3)
        x5 = self.layers[4](x+x1+x2+x3+x4)
        x6 = self.layers[5](x+x1+x2+x3+x4+x5)
        return x6

    def DFF(self, x):  # 深度特征融合模块
        # x1 = self.conv1(x)
        # # 这个3x3的卷积层也可以去掉看看效果
        # x1 = self.conv3(x1)
        x = self.last_deep_Conv(x)
        # 这个CA和PA可以去掉之后再看看
        x = self.PA(x) # 消融实验2
        return x

    def forward(self, x):
        # u = x.clone()
        H, W = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_up = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        x = self.conv_first(x)  # 经过浅层特征提取

        ### 转换到DWT域

        x = self.DFF(self.forward_features(x)) + x  # 经过深层特征提取和特征融合
        # x = self.DFF(self.residual_forward(x)) + x
        # x = self.forward_features(x) + x  # 取消深度特征融合模块

        ### 多尺度池化
        # x = self.PPM(x) (效果不好?)

        x = self.upsample(x) + x_up # 图像上采样重建

        x = x / self.img_range + self.mean

        # ILR = F.interpolate(u, scale_factor=self.scale, mode='bilinear', align_corners=False)
        # x[:, :, :H * self.upscale, :W * self.upscale] + ILR
        return x[:, :, :H * self.upscale, :W * self.upscale]




if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # 720p [1280 * 720]
    # 0.31M | 1.234G | 33.965M time:3-30
    #  0.307M | 1.22G | 33.817M
    # 0.324M  | 1.294G  | 32.834M
    # 0.293M | 1.153G | 27.919M (new)
    # 0.289M | 1.153G| 27.919M (eca)
    # 0.345M | 1.38G | 30.295M (eca+PA+ffn_expand=4+shuffle_C)
    ## 0.25M | 0.998G | 22.037M (计算注意力时通道数不加倍)
    ## 0.279M | 1.111G| 24.396M (new test7)
    ## 0.237M | 0.947G| 20.279M (分成了四部分 pool)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    ##0.125M| 0.504G| 8.188M largeKernel 48c
    ## 0.194M| 0.784G| 10.777M
    ## 0.286M | 1.143G| 19.108M (time: 4/14)
    ## 0.226M | 0.898G | 18.322M (residual)
    ## 4.817M | 4.293G | 48.956M  (unet)
    ## 0.108M | 0.428G| 9.466M (time: 4/15 depth=4)
    ## 0.131M | 0.514G | 9.467M (多尺度池化)
    ## 0.131M | 0.529G | 13.398M (inceptionFormer restormer_swin_light)
    ## 0.103| 0.414G| 9.957M (inception + CAB)
    x = torch.randn(1, 3, 64, 64)

    model = myNet(embed_dim=48)
    # model = unetSR()
    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(f'output: {output.shape}')
    # model = LCB(48)
    # model = myNet(embed_dim=64)
    # # device = torch.device('cuda:0')
    # # input1 = input1.to(device)
    # # model.eval()
    # # model = model.to(device)
    #
    # out = model(input1)
    # print(out.shape)
    #########################
    #######CNN-SR-PoolFormer############
    # FPS: 86.865487
    # avgtime: 0.011512
    # floaps: 2060223936.0
    # params: 504965.0
    ###########################
    #####################################
    ##############(1)###################
    ###########使用BSRB提取局部特征 de CNN-SR-PoolFormer#########
    # floaps: 1532790208.0
    # params: 376197.0
    # memory: 648852480
    ################################
    #############在(1)的基础上将PSAatten替换为ESA
    # floaps: 1138085888.0
    # params: 314433.0
    # memory: 609654272
    #############################
    #######################在(1)的基础上将PSAatten替换为CCA##(2)##
    # floaps: 1089243392.0
    # params: 270033.0
    # memory: 650612224
    ######################################
    ########################################
    ###########使用可变卷积和maxPool共同提取全局特征######
    # floaps: 1393330432.0
    # params: 343065.0
    # FPS: 11.142584
    # avgtime: 0.089746
    # memory: 3828447232
    ###############################################
    ###############################################
    ############在(2)的基础上将并行提取改为串行提取##(3)#####
    # 将conv1提到模块开头##################
    # FPS: 99.202814
    # avg time: 0.010080
    # floaps: 1022134528.0
    # params: 253649.0
    ##############################################
    ####### 在(3)的基础上将embed_dim 设置为 48 ，暂定为该版本为基准
    # floaps: 599483712.0
    # params: 149065.0
    # memory: 480545792
    # FPS: 114.247583 avg time: 0.008753
    ###############################################(4)###
    # 在(3)的基础上将PoolFormer改为PoolFormerT
    # floaps: 1559005440.0
    # params: 386001.0
    #################################
    #####################################
    #在(4)的基础上将embed_dim = 64 改为 48
    # floaps: 901473600.0
    # params: 223753.0
    ###################################
    ########## 在(3)的基础上去除了CCA层之前的conv1#####
    # floaps: 955025664.0
    # params: 237009.0
    #############################
    #########在(3)的基础上将PoolFormer改成Criss-cross Atten
    # floaps: 508315904.0
    # params: 125457.0
    # memory: 640964096
    ################################
    ###############################
    ##########在(3)的基础上将CCA替换为CoordAtt#######
    # floaps: 956628992.0
    # params: 241377.0
    # memory: 621565440
    #############################################

    #################LCB+BSRB#####################
    # floaps: 336815424.0
    # params: 84361.0
    # memory: 267953152
    ######## 在LCB内部增加一层Conv1
    # floaps: 349398336.0
    # params: 87241.0
    ####### LKA + 局部注意力 time:3-21
    # floaps: 897124320.0
    # params: 222354.0
    #######3 使用restormer结构 time:3-22
    # floaps: 939147264.0
    # params: 229428.0
    #######3 基于ShuffleMixer修改 depth=6 dim=64 time:3-27
    # floaps: 1375476480.0
    # params: 347936.0
    # 0.349M | 1.376G | 24.315M
    ######### new dim=48
    # 0.307M | 1.209G | 30.082M
    #######################
    # 0.296M | 1.179G | 33.01M
    # floaps, params = profile(model, inputs=(input1,))
    # print('floaps: ', floaps)
    # print('params: ', params)
