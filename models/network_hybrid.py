# 测试
# 开发时间：2022/8/5 9:26
from thop import profile
# import torchvision.models as models
# import torch
from ptflops import get_model_complexity_info
from models.basicblock import DRB, BSConvURB, PALayer, CALayer,CCALayer
from models.SwinT import SwinT
# from .FCVit import fcvit_block
# from .FCA import MultiSpectralAttentionLayer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode
from models.fusion import iAFF ,AFF, MS_CAM

# 双三次上采样
# img = trans_fn.resize(img, size, InterpolationMode.BICUBIC)

def channel_shuffle(x, groups=4):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups# reshape
    x = x.view(batchsize, groups,channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

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


# 改进1： depth 默认为4 ，现在改为6 (已测试)
# 改进：depth改为6 (已测试)
# upscale = 2 / 4
class myModel(nn.Module):
    def __init__(self, img_size=64, num_heads=8, upscale=4, window_size=8, num_in_ch=3, nf=64, embed_dim=64,
                 depth=4, upsampler='pixelshuffledirect', img_range=1.):
        super(myModel, self).__init__()
        num_feat = 64
        num_out_ch = 3
        self.upsampler = upsampler
        self.window_size = window_size
        self.img_range = img_range
        self.upscale = upscale
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # self.iAff = iAFF(channels=embed_dim)
        # self.Aff = AFF(channels=embed_dim)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        # self.RRDB = RRDB(embed_dim,num_heads)
        # self.firstUp = nn.PixelShuffle(upscale)
        # self.conv_end1 = nn.Conv2d(embed_dim//pow(upscale,2), num_in_ch, kernel_size=1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = depth
        self.layers = nn.ModuleList()  # 存放HRBCT模块
        for i_layer in range(self.num_layers):
            layer = HRBCT(embed_dim, nf, num_heads)
            self.layers.append(layer)

        #####################################################################################################
        ################################### 2.2, 深度特征融合模块 ######################################
        self.conv1 = nn.Conv2d(depth*embed_dim, embed_dim, kernel_size=1) # depth*embed_dim
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)
        # self.CA = CALayer(embed_dim) # 消融实验1
        self.PA = PALayer(embed_dim) # 消融实验2

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (img_size, img_size))

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_shallow_features(self, x):
        x1 = self.RRDB(x)
        x1 = self.firstUp(x1)
        x1 = self.conv_end1(x1)
        return x1

    def forward_features(self, x):  # 经过深层特征提取（HRBCT）之后的输出
        # 经过HRBCT之后的x，其shape未发生改变

        # 可以考虑将浅层特征输入到每一层核心模块中(已测试)(可注销)
        # st = x # 保存浅层特征
        # index = 0
        # retainV = []
        # for layer in self.layers:
        #     index += 1
        #     if index > 1:
        #         x = layer(x + st)  # 之后也可以换成 concat + conv1
        #     else:
        #         x = layer(x)
        #     retainV.append(x)


        retainV = []
        for layer in self.layers:
            x = layer(x)
            retainV.append(x)

        # depth=2
        # x1 = torch.cat((retainV[0], retainV[1]),1).contiguous()

        # depth=4 改进2：将每个HRBCT模块的输出 concat到一起，然后输入到深度特征融合层
        x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3]), 1).contiguous()

        # depth=6
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5]), 1).contiguous()

        # depth=8
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5],retainV[6],retainV[7]), 1).contiguous()

        # depth=10
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5],retainV[6],retainV[7],retainV[8],retainV[9]), 1).contiguous()

        # depth=12
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5],retainV[6],retainV[7],retainV[8],retainV[9],retainV[10],retainV[11]), 1).contiguous()

        return x1

    def DFF(self, x):  # 深度特征融合模块
        x1 = self.conv1(x)
        # 这个3x3的卷积层也可以去掉看看效果
        x1 = self.conv3(x1)
        # 这个CA和PA可以去掉之后再看看
        # x1 = self.CA(x1)  # 消融实验1
        x1 = self.PA(x1) # 消融实验2
        return x1

    def forward(self, x):
        H, W = x.shape[2:]
        # x = self.check_image_size(x)

        # 先将LR输入进行双三次上采样(已测试)
        # SR = trans_fn.resize(x, (self.upscale*H,self.upscale*W), InterpolationMode.BICUBIC)

        # 效果不好的话，这段代码也可以考虑取消
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # 默认采用第二种轻量级上采样方式
        # 上采样时，可以考虑加入 PA 或 CA + PA 模块
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            # print('卷积前的大小：', x.shape)  # 3 120 200
            x = self.conv_first(x)

            # x = self.DFF(self.forward_features(x)) + x

            # 使用 iAFF融合特征
            # x = self.Aff(self.DFF(self.forward_features(x)),x)

            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR

            # SR = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

            x = self.conv_first(x)  # 经过浅层特征提取

            # 将浅层特征直接上采样，然后再与最后的输出求和(已测试)
            # out1 = self.upsample(x)

            #（yi测试） 多分出一条路，将浅层特征提取之后的特征经过RRDB之后放大回原来大小，在将此结果添加到主线输出的结果上
            # end1 = self.forward_shallow_features(x)

            # 也可以考虑将浅层特征x送入每一层核心模块
            # 改进3：调用函数的形似实现
            x = self.DFF(self.forward_features(x)) + x  # 经过深层特征提取和特征融合

            # basic版本  经过多个HRBCT构建好的深度特征提取层后，x的形状未发生改变
            # x = self.forward_features(x) + x  # 经过深层特征提取

            x = self.upsample(x)  # 图像上采样重建

            # 将浅层特征直接上采样之后的结果添加到主线输出的结果上(已测试)(可注销)
            # x = x + out1
        # 将LR双三次上采样之后的特征图直接添加到最终输出结果上(已测试)(可注销)
        # x = x + SR
        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]



# nf的值应该是与embed_dim一致的，embed_dim和nf的值也可以考虑设为64
# 可以考虑添加一个CA模块
# 添加了蒸馏模块，distillation_rate=0.25 或则 0.50
class HRBCT(nn.Module):
    def __init__(self, embed_dim=64, nf=64, num_heads=8,distillation_rate=0.50):
        super(HRBCT, self).__init__()
        # 知识蒸馏
        self.distilled_channels = int(embed_dim * distillation_rate)
        self.remaining_channels = int(embed_dim - self.distilled_channels)
        self.distillation_rate = distillation_rate
        # self.act = nn.Mish()

        # 并行蒸馏(已测试 0.50)
        # self.ParaConv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # self.conv1 = nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1)
        # self.conv2 = nn.Conv2d((embed_dim // 2)*3, embed_dim, kernel_size=1)

        # 测试(给蒸馏出的特征添加conv1)
        # self.conv1_D1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, kernel_size=1)
        # self.conv1_D2 = nn.Conv2d(int(self.remaining_channels * self.distillation_rate), int(self.remaining_channels * self.distillation_rate), kernel_size=1)
        # self.conv1_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # 把Conv1替换为Conv3
        self.Conv3_D1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, 1, 1)
        self.Conv3_D2 = nn.Conv2d(int(self.remaining_channels * self.distillation_rate), int(self.remaining_channels * self.distillation_rate), 3, 1, 1)

        # self.conv1_ST = nn.Conv2d(self.remaining_channels, self.remaining_channels, kernel_size=1)
        # SwinT不改变输入特征图的尺寸，只改变通道数
        self.ST = SwinT(embed_dim=self.remaining_channels, heads=num_heads)
        # 替换为FCVit
        # self.FCVitB = fcvit_block()


        # 密集残差块
        # self.RDB = RDB(nf=int(nf*(1-distillation_rate)**2), gc=int((nf*(1-distillation_rate)**2)//2))  # 输入通道和中间层输出通道 可考虑设为30

        # 并行蒸馏(已测试)
        # self.SRB = SRB(self.distilled_channels)
        # self.conv1_SRB = nn.Conv2d(int(nf*(1-distillation_rate)**2), int(nf*(1-distillation_rate)**2), kernel_size=1)

        # 替换为浅层残差块，目前结构确定使用的模块
        # self.SRB = SRB(int(nf*(1-distillation_rate)**2))

        # 使用BSConvURB
        # self.BSRB = BSConvURB( int(nf*(1-distillation_rate)**2), int(nf*(1-distillation_rate)**2), kernel_size=3)

        # DRB
        # self.DRB = DRB(int(nf*(1-distillation_rate)**2))

        # ESA也是一样
        # self.ESA = ESA(n_feats=nf, conv=nn.Conv2d)  # 输出通道 输入通道

        # 添加通道注意力机制
        # self.CA = CALayer(nf)

        # 添加 CCALayer注意力机制
        self.CCA = CCALayer(nf)

        # self.pixel_norm = nn.LayerNorm(nf)
        # iAFF
        # self.iAff = iAFF(channels=embed_dim)
        # self.Aff = AFF(channels=embed_dim)
        # 添加FCA（频率通道注意力机制）
        # self.FCA = MultiSpectralAttentionLayer()

    def forward(self, x):
        # 改进6：添加CA注意力机制 (已测试 pass)
        # x = self.CA(x)

        # x = self.ST(x)
        # x = self.RDB(x)
        # # CA模块也可以考虑添加到这个位置 (已测试)
        # # 改进8：取消CA模块 (已测试)
        # # 改进10：取消CA模块 保留ESA模块
        # x = self.CA(x)
        # # 改进7：取消ESA模块 (已测试)
        # # x = self.ESA(x)
        # return x
        # 在内部加一个残差连接 (已测试)
        # 在内部加一个经过conv1之后的残差连接
        # x2 = self.conv1(x)

        # 知识蒸馏
        # channel_shuffle
        # x1 = self.conv1_out(x)
        # x1 = channel_shuffle(x)
        distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)

        # 给蒸馏出的特征添加conv1
        # distilled_c1 = self.conv1_D1(distilled_c1)
        # 给蒸馏出的特征添加conv1替换为Conv3
        distilled_c1 = self.Conv3_D1(distilled_c1)

        out1 = self.ST(remaining_c1)
        # out1 = self.FCVitB(remaining_c1) # 替换为FCVit

        # out = torch.cat([distilled_c1, out1], dim=1)

        # 把SRB去除
        # channel_shuffle
        # out1 = channel_shuffle(out1)
        # distilled_c2, remaining_c2 = torch.split(out1, (int(self.remaining_channels*self.distillation_rate), int(self.remaining_channels*(1-self.distillation_rate))), dim=1)
        #
        # # 给蒸馏出的特征添加conv1
        # # distilled_c2 = self.conv1_D2(distilled_c2)
        # # 给蒸馏出的特征添加conv1替换为Conv3
        # distilled_c2 = self.Conv3_D2(distilled_c2)
        #
        # # out2 = self.SRB(remaining_c2)
        # # 使用BSConvU
        # out2 = self.BSRB(remaining_c2)
        # DRB
        # out2 = self.DRB(remaining_c2)

        # out = torch.cat([distilled_c1, distilled_c2, out2], dim=1)
        out = torch.cat([distilled_c1,  out1], dim=1)
        # 以上------------------------------------

        #-----------------------------------------------------------

        # ESA暂时去除
        # x1 = self.ESA(out)

        # 并行蒸馏(已测试)
        # distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)
        # out1 = self.ST(remaining_c1)
        # out2 = self.SRB(distilled_c1)
        # out = torch.cat([out1, out2], dim=1)
        # out = self.ParaConv1(out)
        #-----------------------------------------

        # 替换成 CCA
        # out = self.conv1_out(out)
        x1 = self.CCA(out) # 消融 除去CA

        x_4 = x + x1

        # 添加 pixel_norm
        # x_4 = x_4.permute(0, 2, 3, 1)  # (B, H, W, C)
        # x_4 = self.pixel_norm(x_4)
        # x_4 = x_4.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return x_4


        # x1 = self.ST(x)
        # x1 = self.RDB(x1)
        # # x1 = self.CA(x1)
        # x1 = self.ESA(x1)
        # # x1 = self.FCA(x1)
        # return x2 + x1
        # return x2 + x1*0.2

        # 结合每一层的输出
        # x1 = self.ST(x)
        # x2 = self.conv1(x)
        # x3 = self.conv1(x1)
        # x1 = self.RDB(x1)
        # x4 = self.conv1(x1)
        # x5 = torch.cat((x2, x3, x4), 1).contiguous()
        # x6 = self.conv2(x5)
        # x6 = self.ESA(x6)
        # 使用 iAFF特征融合
        # x7 = self.Aff(x6,x)
        # return x7

        # x6*0.2 + x
        # return x6 + x

# class HRBCT(nn.Module):
#     def __init__(self, embed_dim=64, nf=64, num_heads=8,distillation_rate=0.50):
#         super(HRBCT, self).__init__()
#         # 知识蒸馏
#         self.distilled_channels = int(embed_dim * distillation_rate)
#         self.remaining_channels = int(embed_dim - self.distilled_channels)
#         self.distillation_rate = distillation_rate
#         # self.act = nn.Mish()
#
#         # 并行蒸馏(已测试 0.50)
#         # self.ParaConv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
#
#         # self.conv1 = nn.Conv2d(embed_dim, embed_dim//2, kernel_size=1)
#         # self.conv2 = nn.Conv2d((embed_dim // 2)*3, embed_dim, kernel_size=1)
#
#         # 测试(给蒸馏出的特征添加conv1)
#         # self.conv1_D1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, kernel_size=1)
#         # self.conv1_D2 = nn.Conv2d(int(self.remaining_channels * self.distillation_rate), int(self.remaining_channels * self.distillation_rate), kernel_size=1)
#         # self.conv1_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
#
#         # 把Conv1替换为Conv3
#         self.Conv3_D1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, 1, 1)
#         self.Conv3_D2 = nn.Conv2d(int(self.remaining_channels * self.distillation_rate), int(self.remaining_channels * self.distillation_rate), 3, 1, 1)
#
#         # self.conv1_ST = nn.Conv2d(self.remaining_channels, self.remaining_channels, kernel_size=1)
#         # SwinT不改变输入特征图的尺寸，只改变通道数
#         self.ST = SwinT(embed_dim=embed_dim, heads=num_heads)
#         # 替换为FCVit
#         # self.FCVitB = fcvit_block()
#
#
#         # 密集残差块
#         # self.RDB = RDB(nf=int(nf*(1-distillation_rate)**2), gc=int((nf*(1-distillation_rate)**2)//2))  # 输入通道和中间层输出通道 可考虑设为30
#
#         # 并行蒸馏(已测试)
#         # self.SRB = SRB(self.distilled_channels)
#         # self.conv1_SRB = nn.Conv2d(int(nf*(1-distillation_rate)**2), int(nf*(1-distillation_rate)**2), kernel_size=1)
#
#         # 替换为浅层残差块，目前结构确定使用的模块
#         # self.SRB = SRB(int(nf*(1-distillation_rate)**2))
#
#         # 使用BSConvURB
#         self.BSRB = BSConvURB( embed_dim, embed_dim, kernel_size=3)
#
#         # DRB
#         # self.DRB = DRB(int(nf*(1-distillation_rate)**2))
#
#         # ESA也是一样
#         # self.ESA = ESA(n_feats=nf, conv=nn.Conv2d)  # 输出通道 输入通道
#
#         # 添加通道注意力机制
#         # self.CA = CALayer(nf)
#
#         # 添加 CCALayer注意力机制
#         self.CCA = CCALayer(nf)
#
#         # self.pixel_norm = nn.LayerNorm(nf)
#         # iAFF
#         # self.iAff = iAFF(channels=embed_dim)
#         # self.Aff = AFF(channels=embed_dim)
#         # 添加FCA（频率通道注意力机制）
#         # self.FCA = MultiSpectralAttentionLayer()
#
#     def forward(self, x):
#         # 改进6：添加CA注意力机制 (已测试 pass)
#         # x = self.CA(x)
#
#         # x = self.ST(x)
#         # x = self.RDB(x)
#         # # CA模块也可以考虑添加到这个位置 (已测试)
#         # # 改进8：取消CA模块 (已测试)
#         # # 改进10：取消CA模块 保留ESA模块
#         # x = self.CA(x)
#         # # 改进7：取消ESA模块 (已测试)
#         # # x = self.ESA(x)
#         # return x
#         # 在内部加一个残差连接 (已测试)
#         # 在内部加一个经过conv1之后的残差连接
#         # x2 = self.conv1(x)
#
#         # 知识蒸馏
#         # channel_shuffle
#         # x1 = self.conv1_out(x)
#         # x1 = channel_shuffle(x)
#         # distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)
#
#         # 给蒸馏出的特征添加conv1
#         # distilled_c1 = self.conv1_D1(distilled_c1)
#         # 给蒸馏出的特征添加conv1替换为Conv3
#         # distilled_c1 = self.Conv3_D1(distilled_c1)
#
#         out1 = self.ST(x)
#         # out1 = self.FCVitB(remaining_c1) # 替换为FCVit
#
#         # out = torch.cat([distilled_c1, out1], dim=1)
#
#         # 把SRB去除
#         # channel_shuffle
#         # out1 = channel_shuffle(out1)
#         # distilled_c2, remaining_c2 = torch.split(out1, (int(self.remaining_channels*self.distillation_rate), int(self.remaining_channels*(1-self.distillation_rate))), dim=1)
#
#         # 给蒸馏出的特征添加conv1
#         # distilled_c2 = self.conv1_D2(distilled_c2)
#         # 给蒸馏出的特征添加conv1替换为Conv3
#         # distilled_c2 = self.Conv3_D2(distilled_c2)
#
#         # out2 = self.SRB(remaining_c2)
#         # 使用BSConvU
#         out2 = self.BSRB(out1)
#         # DRB
#         # out2 = self.DRB(remaining_c2)
#
#         # out = torch.cat([distilled_c1, distilled_c2, out2], dim=1)
#         # 以上------------------------------------
#
#         #-----------------------------------------------------------
#
#         # ESA暂时去除
#         # x1 = self.ESA(out)
#
#         # 并行蒸馏(已测试)
#         # distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)
#         # out1 = self.ST(remaining_c1)
#         # out2 = self.SRB(distilled_c1)
#         # out = torch.cat([out1, out2], dim=1)
#         # out = self.ParaConv1(out)
#         #-----------------------------------------
#
#         # 替换成 CCA
#         # out = self.conv1_out(out)
#         x1 = self.CCA(out2) # 消融 除去CA
#
#         x_4 = x + x1
#
#         # 添加 pixel_norm
#         # x_4 = x_4.permute(0, 2, 3, 1)  # (B, H, W, C)
#         # x_4 = self.pixel_norm(x_4)
#         # x_4 = x_4.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
#
#         return x_4

if __name__ == '__main__':
    x = torch.randn((1, 3, 64, 64))
    model = myModel()
    y = model(x)
    print(y.shape)
    device = torch.device('cuda:0')
    input = x.to(device)
    model.eval()
    model = model.to(device)
    macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # floaps, params = profile(model, inputs=(input,))
    # print('floaps: ', floaps)
    # print('params: ', params)
    # block=6
    # 1.08GMac 291.48 k
    # block=2
    # 542.02MMac 141.83k epoch:1000, iter: 200,000, Average PSNR : 39.24dB, Average SSIM : 0.9257

    # block=8
    # 1.34GMac 366.31 k epoch:995, iter: 199,000, Average PSNR : 39.47dB, Average SSIM : 0.9272
    # block=10
    # 1.61GMac 441.13 k
    # block=12
    # 1.88GMac 515.95 k
    # noSplit block=4
    # 1.5GMac 433.62 k
    # block=4
    # 216.66K
    # no CCA
    # Computational complexity: 808.49MMac
    # Number of parameters: 214.34k

    # no RDB
    # Computational complexity:       764.72 MMac
    # Number of parameters:           214.99 k
