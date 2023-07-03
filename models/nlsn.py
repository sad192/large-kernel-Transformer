from models import common
from models import attention
import torch.nn as nn

def make_model():

        return NLSN()


class NLSN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(NLSN, self).__init__()

        n_resblock = 32  # 20
        n_feats = 256  # 64
        kernel_size = 3 
        scale = 2  # 4 2
        act = nn.ReLU(True)
# --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model NLSN --scale 4 --patch_size 96
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=144, n_hashes=4, reduction=4, res_scale=0.1)]

        for i in range(n_resblock):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ))
            if (i+1)%8==0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=144, n_hashes=4, reduction=4, res_scale=0.1))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, 3, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

