# 测试
# 开发时间：2022/7/11 16:27
import math

import torch
from torch import nn

import utils.option as option
import utils.utils_image as util
from data import dataset_deg
from torch.utils.data import DataLoader
from utils.deg_arch import DegModel
from utils.discriminator import NLayerDiscriminator as patchGANLoss
from torch.optim import lr_scheduler
from torch.optim import Adam
from tensorboardX import SummaryWriter

def optimize_parameters(train_data,current_step ,device,G_optimizer,D_optimizer, model, net_D, loss):
    data_H = train_data['H'].to(device)
    data_T = train_data['T'].to(device)

    # 先更新判别器
    for p in model.parameters():
        p.requires_grad = False
    D_optimizer.zero_grad()

    D_loss = 0
    pred_d_real = net_D(data_T)  # 1) real data

    G_data_L = model(data_H)



def main():
    device = torch.device('cuda:0')
    # 获取配置文件
    opt = option.parse('../options/2022deg.yml')
    path_opt = {}  # 存放H和T图片的路径
    # 定义数据集
    dataloader_batch_size = 4
    train_set = dataset_deg(path_opt)
    train_size = int(math.ceil(len(train_set) / dataloader_batch_size))
    print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
    train_loader = DataLoader(train_set,
                              batch_size=dataloader_batch_size,
                              shuffle=True,
                              num_workers=6,
                              drop_last=True,
                              pin_memory=True)
    # 参数为：下采样倍数 图片通道数 模糊核模型配置选项 噪声核配置选项
    model = DegModel(4, 3, opt['setting']['kernel_opt'], opt['setting']['noise_opt'])
    model.to(device)
    # 定义判别器
    net_D = patchGANLoss
    net_D.to(device)
    # 定义损失函数
    # loss
    BCE_loss = nn.BCELoss().cuda()
    L1_loss = nn.L1Loss().cuda()

    # --------------
    # D_result = D(x_, y_).squeeze()
    # D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))
    #
    # G_result = G(x_)
    # D_result = D(x_, G_result).squeeze()
    # D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))
    #
    # D_train_loss = (D_real_loss + D_fake_loss) * 0.5
    # --------------

    # --------------
    # G_result = G(x_)
    # D_result = D(x_, G_result).squeeze()
    # BCE_loss和L1_loss结合  opt.L1_lambda 默认值为100
    # G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + opt.L1_lambda * L1_loss(G_result,
    #                                                                                                           y_)
    # --------------

    # 定义优化函数
    G_optimizer = Adam(model.parameters(), lr=0.0002, weight_decay=0)
    G_scheduler = lr_scheduler.MultiStepLR(G_optimizer, milestones=[], gamma=0.5)

    D_optimizer = Adam(net_D.parameters(), lr=0.0002, weight_decay=0)
    D_scheduler = lr_scheduler.MultiStepLR(D_optimizer, milestones=[], gamma=0.5)

    current_step = 0
    current_epoch = 0
    for epoch in range(30000):
        current_epoch += 1
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # 更新参数
            optimize_parameters(train_data,current_step,device,G_optimizer,D_optimizer,model,net_D,loss)

