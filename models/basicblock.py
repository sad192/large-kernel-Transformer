import functools
from collections import OrderedDict
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import module_util as mutil
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 12]):
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


'''
VGG descriminator (D) for ESRGAN
Taken from: https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/discriminator_vgg_arch.py
'''
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(2048 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function, negative_slope一般取0.01
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc = 3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
'''

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
# def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
#          negative_slope=0.2):
#     L = []
#     for t in mode:
#         if t == 'C':
#             L.append(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                           padding=padding, bias=bias))
#         elif t == 'T':
#             L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                         stride=stride, padding=padding, bias=bias))
#         elif t == 'B':
#             L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
#         elif t == 'I':
#             L.append(nn.InstanceNorm2d(out_channels, affine=True))
#         elif t == 'R':
#             L.append(nn.ReLU(inplace=True))
#         elif t == 'r':
#             L.append(nn.ReLU(inplace=False))
#         elif t == 'L':
#             L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
#         elif t == 'l':
#             L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
#         elif t == '2':
#             L.append(nn.PixelShuffle(upscale_factor=2))
#         elif t == '3':
#             L.append(nn.PixelShuffle(upscale_factor=3))
#         elif t == '4':
#             L.append(nn.PixelShuffle(upscale_factor=4))
#         elif t == 'U':
#             L.append(nn.Upsample(scale_factor=2, mode='nearest'))
#         elif t == 'u':
#             L.append(nn.Upsample(scale_factor=3, mode='nearest'))
#         elif t == 'v':
#             L.append(nn.Upsample(scale_factor=4, mode='nearest'))
#         elif t == 'M':
#             L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
#         elif t == 'A':
#             L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
#         else:
#             raise NotImplementedError('Undefined type: '.format(t))
#     return sequential(*L)


def conv_dw(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
            negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias, groups=in_channels))
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               groups=1))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# --------------------------------------------
# conditional batch norm
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
# --------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# --------------------------------------------
# Concat the output of a submodule to its input
# --------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


# --------------------------------------------
# sum the output of a submodule to its input
# --------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv_dw(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


# --------------------------------------------
# simplified information multi-distillation block (IMDB)
# x + conv1(concat(split(relu(conv(x)))x3))
# --------------------------------------------
class IMDBlock(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CL',
                 d_rate=0.25, negative_slope=0.05):
        super(IMDBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0],
                            negative_slope=negative_slope)

    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res


# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------
# class CALayer(nn.Module):
#     def __init__(self, channel=64, reduction=16):
#         super(CALayer, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_fc = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_fc(y)
#         return x * y


# --------------------------------------------
# Residual Channel Attention Block (RCAB)
# --------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 reduction=16, negative_slope=0.2):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# --------------------------------------------
# Residual Channel Attention Group (RG)
# --------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 reduction=16, nb=12, negative_slope=0.2):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction, negative_slope)
              for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        res = self.rg(x)
        return res + x


# --------------------------------------------
# Residual Dense Block
# style: 5 convs
# --------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(nc + gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(nc + 2 * gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(nc + 3 * gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv5 = conv(nc + 4 * gc, nc, kernel_size, stride, padding, bias, mode[:-1], negative_slope)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


# --------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# --------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


###### 分离卷积 ############
class DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, p=0, stride=1, dilation=1, bias=False):
        super(DResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=stride, dilation=dilation,
                      groups=in_channels, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            # nn.Tanhshrink(),v3
            # nn.Softsign(), V4

            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=stride, dilation=dilation,
                      groups=in_channels, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, x):
        return x + self.net(x)


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C' + mode,
               negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                    negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R',
                           negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


'''
# --------------------------------------------
# NonLocalBlock2D:
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# --------------------------------------------
'''


# --------------------------------------------
# non-local block with embedded_gaussian
# https://github.com/AlexHex7/Non-local_pytorch
# --------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False,
                 downsample_mode='maxpool', negative_slope=0.2):

        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C' + act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


'''
RRDB Generator -- Taken from https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/RRDBNet_arch.py
This is the G for Proposed GAN
'''


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        # 可以考虑 concat连接后，通过 1X1的卷积直接返回
        # self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 1, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.lrelu = nn.Mish(inplace=True)

        # 设置一个可学习的参数进行残差连接
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.25)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # (应该不需要)也可以考虑将conv5设为1x1的卷积，输入concat聚合后的数据直接返回,
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # return x5
        return x5 * 0.2 + x


# 不改变通道数
class PA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


# 不改变通道数 nn.LeakyReLU(negative_slope=0.2, inplace=True) nn.ReLU(inplace=True)
# nn.GELU()
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            torch.nn.SiLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=48, k_size=3,gamma=2,b=1):
        super(eca_layer, self).__init__()
        t = int(abs((log(channel, 2) + b) / gamma))
        k = t if t%2 else t+1
        # print('k',k)
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x) + self.contrast(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
# contrast-aware channel attention module  reduction默认是16
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CCALayer, self).__init__()
        # nn.ReLU(inplace=True) nn.LeakyReLU(0.2, inplace=True)[默认]
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            torch.nn.SiLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        # y.shape = (2,64,1,1)
        y = self.conv_du(y)
        return x * y

# 不改变通道数
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


##### ConvFFN
class ConvFFN(nn.Module):
    def __init__(self, dim, expand=4, bias=True):
        super(ConvFFN, self).__init__()
        hiddenC = int(dim*expand)
        self.conv1 = nn.Conv2d(dim, hiddenC, kernel_size=1, bias=bias)
        self.act = nn.SiLU(inplace=True)
        self.dwconv3 = nn.Conv2d(hiddenC, hiddenC, kernel_size=3, padding=1, groups=hiddenC)  # depthwise conv
        self.tras_conv1 = nn.Conv2d(hiddenC, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.act(out1)
        out1 = self.dwconv3(out1) + out1 ## + out1
        out1 = self.tras_conv1(out1)

        ### 已测试
        # out1 = self.dwconv3(x)
        # out1 = self.act(out1)
        # out1 = self.conv1(out1)
        # out1 =self.act(out1)
        # out1 = self.tras_conv1(out1)

        # out1 = self.conv1(x)
        # out1 = self.dwconv3(x)
        # out1 = self.act(out1)
        # out1 = self.tras_conv1(out1)

        return out1
# nn.ReLU(inplace=True) nn.Mish(inplace=True)
class LargeKernelB(nn.Module):
    def __init__(self,dim=64):
        super(LargeKernelB, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=True)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.conv5(x1)
        x1 = self.conv_spatial(x1)
        out = self.act(x1 + x)
        return out

class SRB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        # if bn_kwargs is None:
        #     bn_kwargs = {}

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
        self.act = torch.nn.SiLU(inplace=True) # SiLU

    def forward(self, x):
        fea = self.pw(x)
        fea = self.dw(fea)
        fea = self.act(fea+x)
        return fea

class DRB(nn.Module):
    def __init__(self, dim,kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros"):
        super().__init__()
        # self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=int(2*dim),
            out_channels=dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=bias,
            padding_mode=padding_mode,
        )
        # hybrid使用 torch.nn.GELU()
        # ALL-CNN使用 SiLU(inplace=True)
        self.act = torch.nn.SiLU(inplace=True)  # SiLU
        # self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        # attn = self.pointwise(x)
        x1 = self.depthwise(x)

        x2 = self.pw(torch.cat((x, x1),1))
        x2 = self.dw(x2)

        fea = self.act(u + x1 + x2)
        return fea

# 融合多尺度特征图 h/2,w/2,2c  h/4,w/4,4c
# dim1表示浅层特征的通道数 2c
class fuse_multiple_scale_map(nn.Module):
    def __init__(self, dim1,dim2):
        super(fuse_multiple_scale_map,self).__init__()
        # 先对小尺寸特征图(深层特征)进行上采样
        # self.conv1 = nn.Conv2d(dim, dim, 3, 1,1)
        self.up = nn.Sequential(nn.Conv2d(dim2, dim2 * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv3 = nn.Conv2d(dim1, dim1, 3, 1, 1)
        ##########
        self.act = torch.nn.SiLU(inplace=True)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(dim1, 4 * dim1,kernel_size=1)
        self.fc2 = nn.Conv2d(4 * dim1, dim1,kernel_size=1)

    def forward(self, m1,m2):
        # u = m2.clone()
        ### m1表示浅层特征(大尺寸)，m2深层特征(小尺寸)
        m2_up = self.up(m2)
        m2 = self.conv3(m2_up)
        m2 = self.act(m2)

        tem1 = m2*m1
        x_out1 = m1+tem1

        x3 = self.max_pool(m2_up)
        x3 = self.fc1(x3)
        x3 = self.act(x3)
        x3 = self.fc2(x3)
        x_out2 = x_out1*x3 + m2_up + x_out1

        return x_out2

#  Large Kernel Channel Attention
class LCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        ## torch.nn.SiLU(inplace=True), nn.Sigmoid()
        self.act1 = torch.nn.SiLU(inplace=True)
        self.conv_du_1 = nn.Sequential(
            nn.Conv2d(channel, channel , 1, padding=0, bias=True),
            self.act1,
            nn.Conv2d(channel , channel,kernel_size=3, padding=1, groups=channel,bias=True),
        )
        self.conv_re_1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            self.act1,
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
        )

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def forward(self, x):
        # u = x.clone()
        B, C, H, W = x.shape

        attn1 = self.conv_du_1(x)
        attn1 = attn1.reshape(B, C, H * W)

        attn2 = self.conv_re_1(x)
        # attn2 = rearrange(attn2, 'b c h w -> b c (h w) 1')
        attn2 = attn2.reshape(B, 1,H*W)

        # [N, IC, 1]
        attn = torch.matmul(attn1, attn2.transpose(1,2)) #######(attn1 @ attn2)
        attn = attn.softmax(dim=-2)
        # [N, IC, 1, 1]
        attn = attn.unsqueeze(-1)
        # attn = attn.reshape(B, C, 1,1)

        out = (attn * x)
        return out

#  Large Kernel Attention
class LKA_ori(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwConv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, 1)
        self.dwConv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_1 = nn.Conv2d(dim, dim, 1)
        self.act1 = torch.nn.SiLU(inplace=True)
        # self.act2 = nn.Sigmoid()
        # self.conv1_3 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.dwConv3(x)
        attn = self.dwConv5(attn)
        attn = self.conv_spatial(attn)
        #### attn = self.conv1_2(attn)
        attn = self.conv1_1(attn)
        attn = self.act1(attn)
        # attn = self.act2(attn)
        # attn = self.conv1_3(attn)
        return u * attn

# 集成LKA的attention
class LKAAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, kernel_size=1) ##2*d_model
        self.activation = torch.nn.SiLU(inplace=True) #####nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class LKAAttention_ori(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = torch.nn.SiLU(inplace=True) #####nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

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


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
# heads = [1,2,4,8]
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
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

###### 向量压缩
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x

#### 双轴注意力
class DualAixAtten(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(DualAixAtten, self).__init__()
        self.conv_q = SRB(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=bias)

        self.conv_k = SRB(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=bias)

        self.conv_v = SRB(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=bias)



        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        print('x:',x.shape)
        # x1.mean(-1).reshape(B, C, H, -1)
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        ## 对k在横纵轴上压缩
        k_copy = k.clone()
        k_row = k.mean(-1).reshape(B, C, H, -1)
        k_col = k_copy.mean(-2).reshape(B, C, -1, W)

        print('kshape:',k_col.shape)
        print('qshape:', q.shape)
        attn_1 = (k_col @ q) ##* self.temperature
        attn_2 = (k_col @ q.transpose(-2, -1))

        attn_3 = (q @ k_row)  ##* self.temperature
        attn_4 = (q.transpose(-2, -1) @ k_row)

        attn = attn_1 + attn_2 + attn_3 + attn_4

        attn = attn.softmax(dim=-1)

        out = (attn * v)

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# 考虑大核卷积在注意力矩阵中的应用
class Conv_Attention(nn.Module):
    def __init__(self, dim,bias=False):
        super(Conv_Attention, self).__init__()
        # self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_q = SRB(in_channels=dim,out_channels=dim,kernel_size=7,padding=3,bias=bias)

        self.conv_k = SRB(in_channels=dim,out_channels=dim,kernel_size=7,padding=3,bias=bias)

        self.conv_v = nn.Sequential(nn.Conv2d(dim, 2*dim, kernel_size=1), nn.Conv2d(2*dim, 2*dim, 3, 1, 1),nn.Conv2d(2*dim, dim, kernel_size=1))


        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # b, c, h, w = x.shape

        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        # q = rearrange(q, 'b (head c) h w -> b (head c) h w', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b (head c) h w', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b (head c) h w', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) # * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn * v)

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# 计算多输入间的权重矩阵
class duoInput_Attention(nn.Module):
    def __init__(self, dim,bias=False):
        super(duoInput_Attention, self).__init__()
        # self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_q = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, (1,7), 1, padding=(0,3)),nn.Conv2d(dim, dim, (7,1), 1, padding=(3,0)))

        self.conv_k = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, (1,7), 1, padding=(0,3)),nn.Conv2d(dim, dim, (7,1), 1, padding=(3,0)))

        self.conv_v = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, (1,7), 1, padding=(0,3)),nn.Conv2d(dim, dim, (7,1), 1, padding=(3,0)))

        self.act = nn.GELU()

        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x1,x2,x3):
        # b, c, h, w = x.shape

        q = self.act(self.conv_q(x1))
        k = self.act(self.conv_k(x2))
        v = self.act(self.conv_v(x3))

        q1 = q.clone()
        k1 = k.clone()
        v1 = v.clone()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)

        attn1 = (q @ k.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out1_1 = (attn1 @ q1)
        attn1 = (q @ v.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out1_2 = (attn1 @ q1)
        out1 = out1_1 + out1_2

        attn2 = (k @ q.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2_1 = (attn2 @ k1)
        attn2 = (k @ v.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2_2 = (attn2 @ k1)
        out2 = out2_1 + out2_2

        attn3 = (v @ q.transpose(-2, -1))
        attn3 = attn3.softmax(dim=-1)
        out3_1 = (attn3 @ v1)
        attn3 = (v @ k.transpose(-2, -1))
        attn3 = attn3.softmax(dim=-1)
        out3_2 = (attn3 @ v1)
        out3 = out3_1 + out3_2

        out1 = self.project_out1(out1) + x1
        out2 = self.project_out2(out2) + x2
        out3 = self.project_out3(out3) + x3

        return out1,out2,out3

# 残差像素注意力
class ResidualPA(nn.Module):
    def __init__(self, dim, hidden_features, bias=True):
        super(ResidualPA, self).__init__()

        # hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=7, stride=1, padding=3,
                                groups=hidden_features * 2, bias=bias)

        self.conv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)
        self.act = torch.nn.SiLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,bias=bias)
        self.sigmoid = nn.Sigmoid()

        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # input = x.clone()
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.act(self.conv1(x1))
        x1 = self.sigmoid(self.conv3(x1))
        x = x1 * x2
        # x = self.project_out(x)
        return x

# dual attention unit
class dualAtten(nn.Module):
    def __init__(self, dim,ffn_expansion_factor=2, bias=True):
        super(dualAtten, self).__init__()
        hidden_feture = int(dim*ffn_expansion_factor)
        self.conv1 = nn.Conv2d(dim, hidden_feture * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(2*hidden_feture, 2*hidden_feture, kernel_size=7, padding=3, groups=2*hidden_feture)  # depthwise conv
        self.CA = PSAModule(hidden_feture,hidden_feture)
        self.PA = ResidualPA(hidden_feture,hidden_feture)
        self.fuse_conv1 = nn.Conv2d(2*hidden_feture, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x1 = self.conv1(x)
        # 划分为两部分，分别计算CA和PA
        x_ca,x_pa = self.dwconv(x1).chunk(2, dim=1)
        x_ca = self.CA(x_ca)
        x_pa = self.PA(x_pa)
        x_fuse = torch.cat([x_ca, x_pa], 1)
        out = self.fuse_conv1(x_fuse)

        return out

# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h = torch.zeros([out_batch, out_channel, out_height,
    #                  out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波 通道数*4 size/2
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

# 提取空间偏置特征，即全局特征
class spacialBias(nn.Module):
    def __init__(self, dim,pool_size=3, bias=True):
        super(spacialBias, self).__init__()
        hidden = 4
        self.conv1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.maxPool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)  # depthwise conv

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxPool(x1)
        x1 = self.dwconv(x1)
        return torch.cat((x,x1),1)



if __name__ == '__main__':
    x1 = torch.randn((2, 48, 64, 64))
    B,C,H,W = x1.shape
    x2 = torch.randn((2, 24, 64, 64))
    x3 = torch.randn((2, 64, 64, 64))
    # B, C, H, W = x1.shape
    # print('x:', x1.shape)
    # x1.mean(-1).reshape(B, C, H, -1)
    # print(x1.reshape(B,C,H*W).shape)
    # esa = ESA(n_feats=60, conv=nn.Conv2d)
    # CA = PALayer(60)
    # rdb = ResidualDenseBlock(60)
    srb = DWT()
    # # # y1,y2,y3 = srb(x1,x2,x3)
    y1 = srb(x1)
    print(y1.shape)
    srb1 = IWT()
    y2 = srb1(y1)
    print(y2.shape)
    # print(int(127.68))

