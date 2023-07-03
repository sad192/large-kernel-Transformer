# 测试
# 开发时间：2022/7/20 10:48
import torch.nn as nn
import torch
from models.network_swinir import SwinIR as swinIRnet
from models.networkEENmodefied import EESN as EENnet

'''
combined EESN
'''


class SwinIR_EESN(nn.Module):
    def __init__(self, opt):
        super(SwinIR_EESN, self).__init__()
        opt_net = opt['netG']
        self.netRG = swinIRnet(upscale=opt_net['upscale'],
                               in_chans=opt_net['in_chans'],
                               img_size=opt_net['img_size'],
                               window_size=opt_net['window_size'],
                               img_range=opt_net['img_range'],
                               depths=opt_net['depths'],
                               embed_dim=opt_net['embed_dim'],
                               num_heads=opt_net['num_heads'],
                               mlp_ratio=opt_net['mlp_ratio'],
                               upsampler=opt_net['upsampler'],
                               resi_connection=opt_net['resi_connection'])
        self.netE = EENnet(up_scale=opt_net['upscale'],
                           img_size=opt_net['img_size'])  # 返回 EESN提取的边缘 和 拉普拉斯提取的边缘

    def forward(self, x):
        x_base = self.netRG(x)  # add bicubic according to the implementation by author but not stated in the paper
        x_learned_lap, x_lap = self.netE(x_base)  # EESN net
        x_finalsr = x_learned_lap + x_base - x_lap
        # 返回ISR, ISR加上经过EESN提取的边缘信息的SR，EESN提取的边缘信息，拉普拉斯提取的边缘
        return x_base, x_finalsr, x_learned_lap, x_lap


if __name__ == '__main__':
    m = nn.ReLU()
    x = torch.rand(1, 3, 2, 2)
    print(x)
    print(m(x))
    print(torch.mean(m(x)))
