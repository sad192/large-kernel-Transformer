# 测试
# 开发时间：2022/8/4 20:27
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from thop import profile
from timm.models.layers import to_2tuple, trunc_normal_


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class SwinT(nn.Module):
    def __init__(
            # self, conv, n_feats, kernel_size,
            # bias=True, bn=False, act=nn.ReLU(True)):
            self,  embed_dim=64, heads=8):

        super(SwinT, self).__init__()
        m = []
        depth = 2
        num_heads = heads
        window_size = 8
        resolution = 64
        mlp_ratio = 2.0
        m.append(BasicLayer(dim=embed_dim,
                            depth=depth,
                            resolution=resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            norm_layer=nn.LayerNorm))
        self.transformer_body = nn.Sequential(*m)

    def forward(self, x):
        res = self.transformer_body(x)
        return res

class RRDB(nn.Module):
    def __init__(
            # self, conv, n_feats, kernel_size,
            # bias=True, bn=False, act=nn.ReLU(True)):
            self, embed_dim=64, heads=8):
        super(RRDB, self).__init__()
        m = []
        depth = 6
        num_heads = heads
        window_size = 8
        resolution = 64
        mlp_ratio = 2.0
        m.append(BasicLayer(dim=embed_dim,
                            depth=depth,
                            resolution=resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            norm_layer=nn.LayerNorm))
        self.transformer_body = nn.Sequential(*m)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):
        res = self.transformer_body(x)
        res = self.conv(res)
        res = res + x
        return res


class BasicLayer(nn.Module):
    def __init__(self, dim, resolution, embed_dim=50, depth=2, num_heads=8, window_size=8,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, norm_layer=None):

        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.window_size = window_size
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, resolution=resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.patch_embed = PatchEmbed(
            embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, h, w

    def forward(self, x):
        x, h, w = self.check_image_size(x)
        _, _, H, W = x.size()
        x_size = (H, W)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.patch_unembed(x, x_size)
        if h != H or w != W:
            x = x[:, :, 0:h, 0:w].contiguous()
        return x

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, pretrained_window_size=0,drop=0.,attn_drop=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.resolution = to_2tuple(resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        # self.attn = WindowAttention_v2(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        #     pretrained_window_size=to_2tuple(pretrained_window_size))

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        # x = x + self.mlp(x)
        return x

# class WindowAttention_v2(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#         pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
#     """
#
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
#                  pretrained_window_size=[0, 0]):
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.pretrained_window_size = pretrained_window_size
#         self.num_heads = num_heads
#
#         self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
#
#         # mlp to generate continuous relative position bias
#         self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
#                                      nn.ReLU(inplace=True),
#                                      nn.Linear(512, num_heads, bias=False))
#
#         # get relative_coords_table
#         relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
#         relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
#         relative_coords_table = torch.stack(
#             torch.meshgrid([relative_coords_h,
#                             relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
#         if pretrained_window_size[0] > 0:
#             relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
#         else:
#             relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
#         relative_coords_table *= 8  # normalize to -8, 8
#         relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
#             torch.abs(relative_coords_table) + 1.0) / np.log2(8)
#
#         self.register_buffer("relative_coords_table", relative_coords_table)
#
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         if qkv_bias:
#             self.q_bias = nn.Parameter(torch.zeros(dim))
#             self.v_bias = nn.Parameter(torch.zeros(dim))
#         else:
#             self.q_bias = None
#             self.v_bias = None
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv_bias = None
#         if self.q_bias is not None:
#             qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
#         qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # cosine attention
#         attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
#         logit_scale = torch.clamp(self.logit_scale,
#                                   max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
#         attn = attn * logit_scale
#
#         relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
#         relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, ' \
#                f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'
#
#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True,
                 qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
            # self.norm2 = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops



class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

if __name__ == '__main__':
    # 输入的通道数要和embed_dim一致
    x = torch.randn((8, 640, 24, 24))
    print(x.shape)
    # x_size = (x.shape[2], x.shape[3])
    model = SwinT(embed_dim=640)
    # 测试模型的大小
    device = torch.device('cpu')
    input = x.to(device)
    model.eval()
    model = model.to(device)
    x = model(input)
    print(x.shape)
