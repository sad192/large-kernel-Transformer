# 测试
# 开发时间：2023/2/27 20:16
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from blindSR.utils import LayerNorm, GRN
from models.basicblock import  BSConvURB, PALayer,CCALayer
from models.CNN_SR import PSAModule
from models.SwinT import SwinT
from models.basicblock import duoInput_Attention,ResidualPA
from models.PoolFormer import PoolFormerBlock
# inplans, planes: 输入通道，输出通道
# 不会改变输入的H和W


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

class LargeKernelConv(nn.Module):
    def __init__(self, dim, kernel_size, small_kernel):
        super(LargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # self.Decom = Decom
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding1 = (kernel_size // 2, small_kernel//2)
        padding2 = (small_kernel//2, kernel_size // 2)
        self.pw1 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.pw2 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.pw3 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.Lconv1 = nn.Conv2d(2*dim, 2*dim, kernel_size=(kernel_size,small_kernel), padding=padding1,stride=1, dilation=1,groups=dim)
        self.Lconv2 = nn.Conv2d(2*dim, 2*dim, kernel_size=(small_kernel,kernel_size), padding=padding2,stride=1, dilation=1,groups=dim)
        self.Sconv = nn.Conv2d(2*dim, 2*dim, kernel_size=(small_kernel,small_kernel), padding=small_kernel//2,stride=1, dilation=1,groups=dim)

        # self.act1 = nn.GELU()  # SiLU
        # self.act2 = nn.GELU()  # SiLU
        # self.act3 = nn.GELU()  # SiLU

    def forward(self, x):
        x1 = self.pw1(x)
        x1 = self.Lconv1(x1)
        # x1 = x+x1
        # x1 = self.act1(x1)

        x2 = self.pw2(x)
        x2 = self.Lconv2(x2)


        x3 = self.pw3(x)
        x3 = self.Sconv(x3)

        return x1 + x2 + x3

# 大核卷积块
class LBlock(nn.Module):
    r""" SLaK Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, large_kernel=7,small_kernel=7):
        super().__init__()

        self.large_kernel = LargeKernelConv(dim=dim,kernel_size=large_kernel,small_kernel=small_kernel)

        self.norm = LayerNorm(2*dim, eps=1e-6)
        self.pwconv1 = nn.Linear(2*dim, 8 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(8 * dim, 4*dim)
        self.act2 = nn.GELU()
        self.pwconv3 = nn.Linear(4 * dim, 1 * dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.large_kernel(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.act2(x)
        x = self.pwconv3(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class R_LBlock(nn.Module):
    def __init__(self, dim, depth=4, large_kernel=7,small_kernel=7,useSwin=False):
        super().__init__()
        # build blocks
        # self.blocks = nn.ModuleList([
        #     LBlock(dim=dim,large_kernel=large_kernel,small_kernel=small_kernel)
        #     for i in range(depth)])
        if useSwin:
            self.blocks = nn.Sequential(*[SwinT(dim) for j in range(depth)])
        else:
            self.blocks = nn.Sequential(
                *[LBlock(dim=dim,large_kernel=large_kernel,small_kernel=small_kernel)
                    for i in range(depth)])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        # for blk in self.blocks:
        #     x = blk(x)
        x1 = self.blocks(x)
        x = self.conv(x1) + x
        return x

class HybridBlock(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 可以考虑添加一个Conv1的卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.swinT = SwinT(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x) + x
        x = self.swinT(x) + x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# 改成先全局再局部
class all_local_Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 可以考虑添加一个Conv1的卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.swinT = SwinT(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # x = self.dwconv(x) + x
        x = self.swinT(x) + x

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 可以考虑添加一个Conv1的卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class FBlock(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, useCA=False,useShffule=False,last=False,depth=[2,4,2],drop_path=0.):
        super(FBlock,self).__init__()

        self.useCA = useCA
        self.useShffule = useShffule
        self.islast = last

        if self.islast:
            self.conv_last1 = nn.Conv2d(dim, 3, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(dim, 3, 3, 1, 1)
            self.conv_last3 = nn.Conv2d(dim, 3, 3, 1, 1)
        # self.norm_layer1 = nn.Sequential(
        #     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
        #     nn.Conv2d(dim, dim, kernel_size=1),
        # )
        self.norm_layer1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.block1 = nn.Sequential(*[Block(dim,drop_path) for j in range(depth[0])])
        # *[FBlock(dim=embed_dim, useShffule=True) for j in range(depths[i])]
        # 下采样 2*dim
        self.downConv1 = nn.Sequential(
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 2*dim, kernel_size=2, stride=2),
        )
        ########pixel shuffle
        # m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        # m.append(nn.PixelShuffle(scale))
        if useShffule and not last:
            self.transC1 = nn.Conv2d(2*dim, 4 * dim, kernel_size=1)
            self.pixelShullfe1 = nn.PixelShuffle(2)
        else:
            self.transC1 = nn.Conv2d(2*dim, dim, kernel_size=1)

        self.block2 = nn.Sequential(*[Block(2*dim,drop_path) for j in range(depth[1])])
        # 下采样 4*dim
        self.downConv2 = nn.Sequential(
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(2*dim, 4*dim, kernel_size=2, stride=2),
        )

        if useShffule and not last:
            self.transC2 = nn.Conv2d(4*dim, 4 * 2*dim, kernel_size=1)
            self.pixelShullfe2 = nn.PixelShuffle(2)
        else:
            self.transC2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # self.norm_layer3 = nn.Sequential(
        #     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
        #     nn.Conv2d(dim, dim, kernel_size=1),
        # )
        self.block3 = nn.Sequential(*[HybridBlock(4*dim,drop_path) for j in range(depth[2])])

        # self.conv1 = nn.Conv2d(dim*7, dim, kernel_size=1)

        if useCA:
            self.CA = PSAModule(dim,dim)


    def forward(self, x):
        x1 = self.norm_layer1(x)
        x1 = self.block1(x1)
        x1_down = self.downConv1(x1) # 2 dim

        # x2 = self.norm_layer2(x1_down)
        x2 = self.block2(x1_down)
        x2_down = self.downConv2(x2) # 4 dim

        # x3 = self.norm_layer3(x2_down)
        x3 = self.block3(x2_down)

        # 融合前三个block的输出
        # F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        ############## Sequentially Add
        # 也可以考虑使用注意力机制去融合这些特征
        # 考虑使用 pixelShuffle 进行上采样
        if self.islast:
            ############## sum融合
            x2 = self.transC1(x2)
            x3 = self.transC2(x3)
            x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3_up = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x1_ori = self.conv_last1(x1)
            x2_ori = self.conv_last2(x2_up)
            x3_ori = self.conv_last3(x3_up)
            return x1_ori + x2_ori + x3_ori
        else:
            x3 = self.transC2(x3)
            if self.useShffule:
                x3_up = self.pixelShullfe1(x3)
            else:
                x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = x3_up + x2
            x4 = self.transC1(x4)
            if self.useShffule:
                x4_up = self.pixelShullfe2(x4)
            else:
                x4_up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
            x_out = x4_up + x1

            return x_out  # 内部可以考虑加一个 short skip connection


# 多路分支注意力Block
class PBlock(nn.Module):
    """ 多路分支.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, pool_size=3, useCA=False,usePA=False,drop_path=0.):
        super(PBlock,self).__init__()
        # self.norm_layer1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # self.conv1_first = nn.Conv2d(dim,2*dim,kernel_size=1)
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=2 * dim, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(in_channels=2*dim, out_channels=(dim//2)*3, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=(dim//2)*3, out_channels=dim, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False))

        self.path2 = nn.Sequential(nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=(1, 1),stride=1,padding=0,dilation=1,groups=1,bias=False),
                                   nn.Conv2d(2*dim, 2*dim, kernel_size=7, padding=3, groups=dim),
                                   nn.Conv2d(in_channels=2 * dim, out_channels=(dim // 2) * 3, kernel_size=(1, 1),
                                             stride=1, padding=0, dilation=1, groups=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=(dim // 2) * 3, out_channels=dim, kernel_size=(1, 1), stride=1,
                                             padding=0, dilation=1, groups=1, bias=False)
                                   )

        self.path3 = nn.Sequential(nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=(1, 1),stride=1,padding=0,dilation=1,groups=1,bias=False),
                                   nn.Conv2d(2*dim, 2*dim, kernel_size=(1,13), padding=(0,13//2), groups=dim),
                                   nn.Conv2d(2*dim, 2*dim, kernel_size=(13,1), padding=(13//2,0), groups=dim),
                                   nn.Conv2d(in_channels=2 * dim, out_channels=(dim // 2) * 3, kernel_size=(1, 1),
                                             stride=1, padding=0, dilation=1, groups=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=(dim // 2) * 3, out_channels=dim, kernel_size=(1, 1), stride=1,
                                             padding=0, dilation=1, groups=1, bias=False)
                                   )

        self.path4 = nn.Sequential(nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2),
                                   nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=(1, 1),stride=1,padding=0,dilation=1,groups=1,bias=False))
        self.duoAtten = duoInput_Attention(dim)
        self.conv1_trans = nn.Conv2d(4*dim,dim,kernel_size=1)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.useCA = useCA
        self.usePA = usePA
        if useCA:
            self.CA = PSAModule(dim,dim)
        if usePA:
            self.PA = ResidualPA(dim,dim)

    def forward(self, x):
        input = x
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)

        x1,x2,x3 = self.duoAtten(x1,x2,x3)

        x_all = torch.cat((x1,x2,x3,x4),1).contiguous()
        x_all = self.conv1_trans(x_all) + x

        x_all = x_all.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x_all = self.norm(x_all)
        x_all = self.pwconv1(x_all)
        x_all = self.act(x_all)
        x_all = self.grn(x_all)
        x_all = self.pwconv2(x_all)
        x_all = x_all.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.useCA:
            x_all = self.CA(x_all)
        if self.usePA:
            x_all = self.PA(x_all)

        x = input + self.drop_path(x_all)
        return x

class PBlock_conv(nn.Module):
    """ 多路分支残差块.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim=64, depth=6, drop_path=0.):
        super(PBlock_conv,self).__init__()
        self.basic_block = nn.Sequential(*[PBlock(dim=dim,useCA=True,usePA=True) for j in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        input = x.clone()
        x = self.basic_block(x)
        x = self.conv(x)

        return x + input

# embed_dim 默认为 96
class mySRNet(nn.Module):
    def __init__(self, inc=3,outc=3,upscale=4,num_feat = 64,embed_dim=64,f_groups=4,f_stages=2,Lblocks=6,layers=6,depths=6,large_kernels=[51,49,49,47,47,13],small_kernels=[5,5,5,5,5,3] ,img_size=64,upsampler='pixelshuffle', img_range=1.,norm_layer=GroupNorm):
        super(mySRNet, self).__init__()

        self.upsampler = upsampler
        self.img_range = img_range
        self.upscale = upscale

        if inc == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        #####################################################################################################
        ############上采样重建
        # self.conv_first_up = nn.Conv2d(inc, embed_dim, 3, 1, 1)
        # ######## 提取高频信息
        # self.num_groups = f_groups
        # self.extractFrequenceInfoB = nn.Sequential(
        #     *[PoolFormerBlock(dim=embed_dim, use_layer_scale=False) for j in range(self.num_groups)]
        #     # drop_path=dp_rates[cur + j]
        # )
        # # 在最后输出时卷积
        # self.deepLastConv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        # self.layers = nn.ModuleList()  # 存放深度特征提取模块
        # ######## 一层里面包含两个FBlock,即两个stage
        # self.num_layers = f_stages
        # for i in range(self.num_layers - 1):
        #     layer = nn.Sequential(
        #         *[FBlock(dim=embed_dim, useShffule=True, useCA=(j == f_stages - 1),last=(j == f_stages - 1),
        #                  depth=depths[j]) for j in range(self.num_layers)]  # drop_path=dp_rates[cur + j]
        #     )
        #     self.layers.append(layer)

        ################################### 1, shallow feature extraction ############
        self.conv_first = nn.Conv2d(inc, embed_dim, 3, 1, 1)
        # self.conv_first = nn.Sequential(
        #     nn.Conv2d(inc, embed_dim, 3, 1, 1),
        #     LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        # )
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        # PoolFormerBlock(dim=inc, use_layer_scale=False)
        self.L_blocks = nn.ModuleList()
        self.num_Lblocks = Lblocks
        # self.nums_layer = layers
        for i in range(self.num_Lblocks):
            layer = PBlock_conv(dim=embed_dim,depth=layers)
            self.L_blocks.append(layer)

        #####################################################################################################
        ################################### 2.2, 深度特征融合模块 ######################################
        ####################self.conv1 = nn.Conv2d(depth * embed_dim, embed_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim,bias=True)
        # self.conv1 = nn.Conv2d(embed_dim, 2*embed_dim, kernel_size=1)  # depth*embed_dim

        # self.conv3 = BSConvU(embed_dim, embed_dim)
        self.PA = PALayer(embed_dim)  #
        #####################################################################################################
        ################################ 3, high quality image reconstruction nn.LeakyReLU(inplace=True) ################################
        if self.upsampler == 'pixelshuffle':
            # self.conv_before_upsample1 = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            #                                            nn.GELU())
            # self.upsample1 = Upsample(upscale, num_feat)
            # self.conv_last1 = nn.Conv2d(num_feat, outc, 3, 1, 1)
            #
            # self.conv_before_upsample2 = nn.Sequential(nn.Conv2d(2*embed_dim, num_feat, 3, 1, 1),
            #                                            nn.GELU())
            # self.upsample2 = Upsample(upscale, num_feat)
            # self.conv_last2 = nn.Conv2d(num_feat, outc, 3, 1, 1)
            #
            # self.conv_before_upsample3 = nn.Sequential(nn.Conv2d(4*embed_dim, 4*embed_dim, 3, 1, 1),
            #                                            nn.GELU())
            # self.upsample3 = nn.Sequential( nn.PixelShuffle(2) )
            # self.conv_last3 = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),nn.Conv2d(num_feat, outc, 3, 1, 1))

            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, outc, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, outc,
                                            (img_size, img_size))

    def beforeDeep(self,x):
        pass

    def up_forward_features(self, x):
        u = x.clone()
        # for i in range(self.num_layers):
        #     x = self.layers[i](x)
        x = self.extractFrequenceInfoB(x)

        x = self.deepLastConv(x) + u
        # 多尺度特征提取
        x = self.layers[0](x)

        # depth=6
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5]), 1).contiguous()
        return x

    def forward_features(self, x):  # 经过深层特征提取（HRBCT）之后的输出
        # retainV = []
        # u = x.clone()
        for i in range(self.num_Lblocks):
            x = self.L_blocks[i](x)
        return x

    def DFF(self, x):  # 深度特征融合模块
        # x1 = self.conv1(x)
        # 这个3x3的卷积层也可以去掉看看效果
        x1 = self.conv3(x)
        # 这个CA和PA可以去掉之后再看看
        x1 = self.PA(x1) # 消融实验2
        return x1

    def forward(self, x):
        # u = x.clone()
        H, W = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # 上采样重建分支
        # x_up = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        # x_up = self.conv_first_up(x_up)
        # x_up_ori = self.up_forward_features(x_up)

        ##### 常规重建分支
        x = self.conv_first(x)  # 经过浅层特征提取

        x = self.DFF(self.forward_features(x)) + x  # 经过深层特征提取和特征融合

        # 图像上采样重建
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else :
            x = self.upsample(x)

        # x = x + x_up_ori

        x = x / self.img_range + self.mean

        # ILR = F.interpolate(u, scale_factor=self.scale, mode='bilinear', align_corners=False)
        # x[:, :, :H * self.upscale, :W * self.upscale] + ILR
        return x[:, :, :H * self.upscale, :W * self.upscale]



if __name__ == '__main__':
    input = torch.rand(1, 3, 64, 64)
    model = mySRNet()
    device = torch.device('cpu')
    input = input.to(device)
    model.eval()
    model = model.to(device)
    out = model(input)
    print(out.shape)