# 测试
# 开发时间：2023/1/16 13:38
# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PoolFormer implementation
"""
import os
import copy
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from utils import module_util as mutil

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


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
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}

class LABAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

class LAB(nn.Module):
    def __init__(self, d_model=48, d_atten=64):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.SiLU()
        self.atten_branch = LABAttention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        # self.pixel_norm = nn.LayerNorm(d_model)
        # mutil.initialize_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    MaxPool2d: 考虑使用MaxPool2d
    nn.MaxPool2d(
            pool_size, stride=1, padding=pool_size // 2)
    AvgPool2d：默认为AvgPool2d
    nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
    """

    def __init__(self, inc=96,pool_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(inc,inc,kernel_size=1)

        self.dw_conv3 = nn.Conv2d(inc,inc, kernel_size=3, padding=1,groups=inc)

        # self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(inc,outc, kernel_size=3, padding=1),
        #                            nn.PixelShuffle(3))
        #
        # self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Conv2d(inc, outc, kernel_size=3, padding=1),
        #                            nn.PixelShuffle(3))
        #
        # self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(3), nn.Conv2d(inc, outc, kernel_size=3, padding=1),
        #                            nn.PixelShuffle(3))
        #
        # self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(6), nn.Conv2d(inc, outc, kernel_size=3, padding=1),
        #                            nn.PixelShuffle(3))
        #
        # self.act = nn.SiLU(inplace=True)

        self.pool1 = nn.MaxPool2d(
            pool_size, stride=1, padding=pool_size // 2)
        # self.pool2 = nn.MaxPool2d(
        #     7, stride=1, padding=7 // 2)
        #
        # self.pool3 = nn.MaxPool2d(
        #     11, stride=1, padding=11 // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv3(x)
        x1 = self.pool1(x) - x
        # u = x.clone()
        # x1 = self.pool1(x)
        # x2 = self.pool2(x)
        # x3 = self.pool3(x)
        # x4 = self.pool4(x)

        return x1  # 保留了高频信息
        # return self.pool(x) + x
        # return self.pool(x)


# class Mlp(nn.Module):
#     """
#     含有 dropout 可能会有影响
#     使用SiLU
#     Implementation of MLP with 1*1 convolutions.
#     Input: tensor with shape [B, C, H, W]
#     """
#
#     def __init__(self, in_features, hidden_features=None,
#                  out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
#         self.drop = nn.Dropout(drop)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1)
        # self.fc2 = nn.Conv2d(hidden_features, out_features,kernel_size=3,stride=1,padding=1,groups=in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PoolFormerT(nn.Module):
    def __init__(self,dim,depth=2):

        super(PoolFormerT, self).__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            PoolFormerBlock(dim, pool_size=3, mlp_ratio=4.,
                            act_layer=nn.SiLU, norm_layer=GroupNorm,
                            drop=0., drop_path=0.,
                            use_layer_scale=False, layer_scale_init_value=1e-5)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.SiLU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        ######################################################
        # 使用MaxPooling提取全局高频信息
        self.token_mixer = Pooling(inc=dim,pool_size=pool_size)

        # 使用大核卷积计算注意力，提取全局信息
        # self.token_mixer = LAB()
        ############################################################
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
        #                act_layer=act_layer, drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 pool_size=3, mlp_ratio=4.,
                 act_layer=nn.SiLU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 pool_size=3,
                 norm_layer=GroupNorm, act_layer=nn.SiLU,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}


@register_model
def poolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s24']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_s36(pretrained=False, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    if pretrained:
        url = model_urls['poolformer_s36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_m36(pretrained=False, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def poolformer_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    if pretrained:
        url = model_urls['poolformer_m48']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


if has_mmseg and has_mmdet:
    """
    The following models are for dense prediction based on 
    mmdetection and mmsegmentation
    """


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s12_feat(PoolFormer):
        """
        PoolFormer-S12 model, Params: 12M
        """

        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s24_feat(PoolFormer):
        """
        PoolFormer-S24 model, Params: 21M
        """

        def __init__(self, **kwargs):
            layers = [4, 4, 12, 4]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 31M
        """

        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                layer_scale_init_value=1e-6,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 56M
        """

        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                layer_scale_init_value=1e-6,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m48_feat(PoolFormer):
        """
        PoolFormer-M48 model, Params: 73M
        """

        def __init__(self, **kwargs):
            layers = [8, 8, 24, 8]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                layer_scale_init_value=1e-6,
                fork_feat=True,
                **kwargs)

if __name__ == '__main__':
    input1 = torch.rand(1, 48, 64, 64)
    # model = PoolFormerBlock(dim=48)
    # model = PoolFormerT(dim=3)
    pool = nn.AdaptiveAvgPool2d(3)
    # out = model(input)
    print('out:', pool(input1).shape)
    # device = torch.device('cuda:1')
    # input = input.to(device)
    # model.eval()
    # model = model.to(device)
    # floaps, params = profile(model, inputs=(input,))
    # print('floaps: ', floaps)
    # print('params: ', params)