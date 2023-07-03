# 测试
# 开发时间：2022/7/13 9:15
import functools

import kornia

from utils import module_util as mutil
import torch.nn.functional as F
import torch
from torch import nn

'''
Only EESN
'''


class EESN(nn.Module):
    def __init__(self):
        super(EESN, self).__init__()
        self.beginEdgeConv = BeginEdgeConv()  # Output 64*64*64 input 3*64*64
        self.denseNet = EESNRRDBNet(64, 256, 64,
                                    5)  # RRDB densenet with 64 in kernel, 256 out kernel and 64 intermediate kernel, output: 256*64*64
        self.maskConv = MaskConv()  # Output 256*64*64
        self.finalConv = FinalConv()  # Output 3*256*256

    def forward(self, x):
        x_lap = kornia.laplacian(x, 3)  # see kornia laplacian kernel
        x1 = self.beginEdgeConv(x_lap)
        x2 = self.denseNet(x1)
        x3 = self.maskConv(x1)
        x4 = x3 * x2 + x2
        print('上采样之前：', x4.shape)
        x_learned_lap = self.finalConv(x4)
        # 返回 EESN提取的边缘 和 拉普拉斯提取的边缘
        return x_learned_lap, x_lap


'''
Starting layer before Dense-Mask Branch
'''


class BeginEdgeConv(nn.Module):
    def __init__(self):
        super(BeginEdgeConv, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv_layer4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv_layer5 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(256, 64, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3,
                                  self.conv_layer4, self.conv_layer5, self.conv_layer6], 0.1)

    def forward(self, x):
        x = self.lrelu(self.conv_layer1(x))
        x = self.lrelu(self.conv_layer2(x))
        x = self.lrelu(self.conv_layer3(x))
        x = self.lrelu(self.conv_layer4(x))
        x = self.lrelu(self.conv_layer5(x))
        x = self.lrelu(self.conv_layer6(x))

        return x


'''
Dense sub branch
'''


class EESNRRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(EESNRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.conv_last(self.lrelu(self.HRconv(fea))))

        return out


'''
Second: Mask Branch of two Dense-Mask branch
'''


class MaskConv(nn.Module):
    def __init__(self):
        super(MaskConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3], 0.1)

    def forward(self, x):
        x = self.lrelu(self.conv_layer1(x))
        x = self.lrelu(self.conv_layer2(x))
        x = self.lrelu(self.conv_layer3(x))
        x = torch.sigmoid(x)

        return x


'''
Final conv layer on Edge Enhanced network
'''


class FinalConv(nn.Module):
    def __init__(self):
        super(FinalConv, self).__init__()

        self.upconv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.HRconv(x)))

        return x


'''
RRDB Generator -- Taken from https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/RRDBNet_arch.py
This is the G for Proposed GAN
'''


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


'''Residual in Residual Dense Block'''


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # 残差连接
        return out * 0.2 + x


if __name__ == '__main__':
    test = torch.randn((2, 64, 64, 64))
    test1 = torch.randn((2, 3, 120, 200))
    # conv1 = nn.Conv2d(3, 180, 3, 1, 1)
    # x = conv1(test)
    # model = EESN()
    rrdb = RRDB(64)
    # net = PatchEmbed()
    # net1 = PatchEmbed1()
    # y = net(test1)
    # y1 = net1(test1)
    x = rrdb(test)
    print(x.shape)
