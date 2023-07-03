# 测试
# 开发时间：2023/2/1 17:05
# ------------------------------------------------------------------------------
# Licensed under the CMC.
# Written by Viet Hoang (vhoang20101999@cmc.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Dict, Optional, Tuple, Union

import os
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import numpy as np
from models.lite_hrnet_element import ConvModule, DepthwiseSeparableConvModule

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


# blocks_dict = {
#     'BASIC': BasicBlock,
#     'BOTTLENECK': Bottleneck
# }


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    # assert (num_channels % groups == 0), ('num_channels should be '
    #                                       'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class SpatialWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        # assert mmcv.is_tuple_of(act_cfg, dict)
        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool = GlobalAvgPool2d()
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class GlobalAvgPool2d(nn.Module):
    """
    Reduce mean over last two dimensions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.mean(dim=-1, keepdim=True)
        return x.mean(dim=-2, keepdim=True)


class CrossResolutionWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        # assert mmcv.is_tuple_of(act_cfg, dict)
        branch_channels = [channel // 2 for channel in channels]
        self.channels = branch_channels
        total_channel = sum(branch_channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = []
        # print(len (x))

        for s in x[:-1]:
            # print(s.size())
            ssize = s.size()[-2:]
            kernel = (int(ssize[0] / mini_size[0]), int(ssize[1] / mini_size[1]))
            assert kernel[0] == kernel[1]
            stride = (
            int((ssize[0] - kernel[0]) / (mini_size[0] - 1)), int((ssize[1] - kernel[1]) / (mini_size[1] - 1)))
            # kernel=[3,3]
            # stride=[(ssize[i]-kernel[i])//mini_size[i] for i in range(len(ssize))]
            # print(F.avg_pool2d(s,kernel,stride).size())
            # print((F.adaptive_avg_pool2d(s, mini_size)).size())
            # print(F.avg_pool2d(s, mini_size))
            out.append(F.avg_pool2d(s, kernel, stride))
            # out.append(F.adaptive_avg_pool2d(s, mini_size))
        out = out + [x[-1]]
        # out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):
        # def _inner_forward(x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]

        return out

        # if self.with_cp and x.requires_grad:
        #     out = cp.checkpoint(_inner_forward, x)
        # else:
        #     out = _inner_forward(x)

        # return out


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x):

        # def _inner_forward(x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        out = torch.cat((self.branch1(x1), x2), dim=1)

        out = channel_shuffle(out, 2)

        return out

        # if self.with_cp and x.requires_grad:
        #     out = cp.checkpoint(_inner_forward, x)
        # else:
        #     out = _inner_forward(x)

        # return out


class IterativeHead(nn.Module):

    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):

        # def _inner_forward(x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out

        # if self.with_cp and x.requires_grad:
        #     out = cp.checkpoint(_inner_forward, x)
        # else:
        #     out = _inner_forward(x)

        # return out


class LiteHRModule(nn.Module):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'),
                with_cp=self.with_cp))
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type='ReLU'),
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels[j],
                                out_channels=in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            nn.Upsample(
                                scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[i])))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(
                                        in_channels=in_channels[j],
                                        out_channels=in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        # print(self.num_branches)
        # print(self.module_type)
        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


class LiteHRNet(nn.Module):

    def __init__(self,
                 cfg,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super().__init__()
        self.extra = cfg['MODEL']['EXTRA']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.with_head = self.extra['WITH_HEAD']
        self.kps_head_cfg = self.extra['KEYPOINTS_HEAD']

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['STEM']['STEM_CHANELS'],
            out_channels=self.extra['STEM']['OUT_CHANELS'],
            expand_ratio=self.extra['STEM']['EXPAND_RATIO'],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.stage0_cfg = self.extra['STAGE0']
        num_channels = self.stage0_cfg['NUM_CHANNELS']
        num_channels = [num_channels[i] for i in range(len(num_channels))]
        self.transition0 = self._make_transition_layer([self.stem.out_channels], num_channels)
        self.stage0, num_pre_chanels = self._make_stage(self.stage0_cfg, num_channels)

        self.stage1_cfg = self.extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS']
        num_channels = [num_channels[i] for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(num_pre_chanels, num_channels)
        self.stage1, num_pre_chanels = self._make_stage(self.stage1_cfg, num_channels)

        self.stage2_cfg = self.extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        num_channels = [num_channels[i] for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(num_pre_chanels, num_channels)
        self.stage2, num_pre_chanels = self._make_stage(self.stage2_cfg, num_channels)

        if self.with_head:
            self.head_layer = IterativeHead(
                in_channels=num_pre_chanels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )
        self.final_layer = nn.Conv2d(
            in_channels=self.kps_head_cfg['IN_CHANELS'],
            out_channels=self.kps_head_cfg['OUT_CHANELS'],
            kernel_size=self.kps_head_cfg['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if self.kps_head_cfg['FINAL_CONV_KERNEL'] == 3 else 0
        )

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=num_channels_pre_layer[i],
                                out_channels=num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            nn.BatchNorm2d(num_channels_pre_layer[i]),
                            nn.Conv2d(
                                in_channels=num_channels_pre_layer[i],
                                out_channels=num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    in_channels,
                    multiscale_output=True):
        num_modules = stages_spec['NUM_MODULES']
        num_branches = stages_spec['NUM_BRANCHES']
        num_blocks = stages_spec['NUM_BLOCK']
        reduce_ratio = stages_spec['REDUCE_RATIO']
        with_fuse = stages_spec['WITH_FUSE']
        module_type = stages_spec['MODULE_TYPE']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))
        in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        else:
            logger.info('=> have no pretrained!')

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        y_list = [x]
        x_list = []
        for j in range(self.stage0_cfg['NUM_BRANCHES']):
            if self.transition0[j]:
                if j < len(y_list):
                    x_list.append(self.transition0[j](y_list[j]))
                else:
                    x_list.append(self.transition0[j](y_list[-1]))

        y_list = self.stage0(x_list)

        x_list = []
        for j in range(self.stage1_cfg['NUM_BRANCHES']):
            if self.transition1[j]:
                if j < len(y_list):
                    x_list.append(self.transition1[j](y_list[j]))
                else:
                    x_list.append(self.transition1[j](y_list[-1]))
            else:
                x_list.append(y_list[j])

        y_list = self.stage1(x_list)

        x_list = []
        for j in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition2[j]:
                if j < len(y_list):
                    x_list.append(self.transition2[j](y_list[j]))
                else:
                    x_list.append(self.transition2[j](y_list[-1]))
            else:
                x_list.append(y_list[j])

        y_list = self.stage2(x_list)

        if self.with_head:
            x = self.head_layer(y_list)
        final_output = self.final_layer(x[0])
        return final_output


def get_pose_net(cfg, is_train, **kwargs):
    model = LiteHRNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model

if __name__ == '__main__':
    input1 = torch.rand(8, 48, 64, 64)
    model = CrossResolutionWeighting(48)
    out = model(input1)
    print(out.shape)