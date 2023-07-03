# 测试
# 开发时间：2023/1/30 11:32
import torch
import torch.nn as nn
from thop import profile

from models import rfdn_block as B

def make_model( parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

if __name__ == '__main__':
    input = torch.rand(2, 3, 64, 64)
    model = make_model()
    out = model(input)
    print(out.shape)
    device = torch.device('cuda:0')
    input = input.to(device)
    model.eval()
    model = model.to(device)
    floaps, params = profile(model, inputs=(input,))
    #### SR=4
    # floaps: 3377969664.0
    # params: 433448.0

    ####### SR=2
    # floaps: 3245259264.0
    # params: 417212.0

    print('floaps: ', floaps)
    print('params: ', params)