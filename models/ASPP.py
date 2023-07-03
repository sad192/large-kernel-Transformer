# 测试
# 开发时间：2023/1/31 10:46
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, GroupNorm=GroupNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = GroupNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class LCBP(nn.Module):
    def __init__(self, inplanes = 48, dilations = [1, 2, 3], GroupNorm=GroupNorm):
        super(LCBP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, 48, 1, padding=0, dilation=dilations[0],)
        self.aspp2 = _ASPPModule(inplanes, 48, 3, padding=dilations[1], dilation=dilations[1],)
        self.aspp3 = _ASPPModule(inplanes, 48, 3, padding=dilations[2], dilation=dilations[2],)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)

        return x3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, GroupNorm=GroupNorm):
        super(ASPP, self).__init__()
        if backbone == 'lab':
            inplanes = 48
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 48, 1, padding=0, dilation=dilations[0],)
        self.aspp2 = _ASPPModule(inplanes, 48, 3, padding=dilations[1], dilation=dilations[1],)
        self.aspp3 = _ASPPModule(inplanes, 48, 3, padding=dilations[2], dilation=dilations[2],)
        self.aspp4 = _ASPPModule(inplanes, 48, 3, padding=dilations[3], dilation=dilations[3],)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 48, 1, stride=1, bias=False),
                                             GroupNorm(48),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(240, 48, 1, bias=False)
        self.bn1 = GroupNorm(48)
        self.act = nn.SiLU()
        # self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone='lab', output_stride=16):
    return ASPP(backbone, output_stride)

if __name__ == '__main__':
    input1 = torch.rand(8, 48, 64, 64)
    model = LCBP(48,[1,3,5])
    out = model(input1)
    print(out.shape)