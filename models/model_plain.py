import functools
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn import init

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, fftLoss, EdgeLoss, GANLoss,PerceptualLoss
from models.loss_ssim import SSIMLoss
from tensorboardX import SummaryWriter

from utils.utils_image import tensor2img
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from models.Adan import Adan

class ModelPlain(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.gpu_items = len(opt["gpu_ids"])
        self.opt_train = self.opt['train']  # training option
        self.opt_net = opt['netG']
        self.opt_path = opt['path']
        self.netG = define_G(opt)
        self.netD = False
        if self.opt_train['task_name'] == 'deg':
            from models.basicblock import NLayerDiscriminator as net
            self.netD = net(3, 64)
            self.netD = self.model_to_device(self.netD)
            # self.init_weights(self.netD)

        self.netE = ''
        # self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.netG = self.model_to_device(self.netG)
        if self.opt_path['isNeedLoad']:
            self.load()  # load model
        if self.opt_net['isTrain']:
            self.netG.train()  # set training mode,for BN
        else:
            self.netG.eval()
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log
        # 定义 tesorboardX
        # version = float(torch.__version__[0:3])
        # if version <= 1.1:  # PyTorch 1.1
        #     print('pytorch版本号:' + str(version))
        #     from torch.utils.tensorboard import SummaryWriter
        # else:
        #     print(
        #         'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
        #
        self.tensorboard_logger = SummaryWriter(log_dir='superresolution/tb_logger', comment='PSRGAN')

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        load_path_trainedG = self.opt['path']['trained_netG']
        if self.opt['path']['isNeedLoadPretrainedModel']:
            if load_path_G is not None:
                print('Loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        else:
            if load_path_trainedG is not None:
                print('Loading model for trained_G [{:s}] ...'.format(load_path_trainedG))
                self.load_network(load_path_trainedG, self.netG, strict=self.opt_train['G_param_strict'],
                                  param_key='params')

        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)

            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        # if self.opt_train['E_decay'] > 0:
        #     self.save_network(self.save_dir, self.netE, 'E', iter_label)
        # if self.opt_train['G_optimizer_reuse']:
        #     self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        E_lossfn_type = self.opt_train['E_lossfn_type']
        P_loss_flag = self.opt_train['p_loss']
        if P_loss_flag:
            self.P_loss = PerceptualLoss(device=self.device).to(self.device)

        if E_lossfn_type == 'fft':
            self.E_lossfn = fftLoss().to(self.device)
        elif E_lossfn_type == 'edge':
            self.E_lossfn = EdgeLoss(self.device).to(self.device)

        # 执行概率退化分布模型时的gan_loss
        if self.opt_train['task_name'] == 'deg':
            self.gan_loss = GANLoss('lsgan', 1.0, 0.0).to(self.device)
            self.nosie_loss = nn.MSELoss().to(self.device)

        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.E_lossfn_weight = self.opt_train['E_lossfn_weight']
        self.P_lossfn_weight = self.opt_train['P_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        E_optim_params = []
        for k, v in self.netG.named_parameters():
            # 判断是否对应参数需要更新
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('G_Params [{:s}] will not optimize.'.format(k))
        # if self.opt_train['task_name'] == 'een':
        #     for k, v in self.netEEN.named_parameters():
        #         # 判断是否对应参数需要更新
        #         if v.requires_grad:
        #             E_optim_params.append(v)
        #         else:
        #             print('EEN_Params [{:s}] will not optimize.'.format(k))
        # 根据配置文件的信息去定义优化器及其参数 self.opt_train['G_optimizer_betas']
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=(0.9,0.99),
                                    weight_decay=self.opt_train['G_optimizer_wd'])
            if self.opt_train['task_name'] == 'deg':
                self.D_optimizer = torch.optim.Adam(self.netD.parameters(), lr=1e-4,
                                                    betas=self.opt_train['G_optimizer_betas'],
                                                    weight_decay=self.opt_train['G_optimizer_wd'])
        elif self.opt_train['G_optimizer_type'] == 'Adan':
            self.G_optimizer = Adan(G_optim_params)
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # 因为继承关系，所以可以访问 schedulers数组,添加的元素为lr_scheduler.MultiStepLR
    # 配置 lr_scheduler，使其根据 epoch 来调整学习率
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
            if self.opt_train['task_name'] == 'deg':
                self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                                self.opt_train['G_scheduler_milestones'],
                                                                self.opt_train['G_scheduler_gamma']
                                                                ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                                            T_0=20,
                                                                            T_mult=2,
                                                                            eta_min = 2e-5
                                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):

        if self.opt_net['net_type'] == 'swinireen':
            self.x_base, self.x_finalSR, self.x_learned_lap, self.x_lap = self.netG(self.L)

        elif self.opt_net['net_type'] == 'deg':
            # , self.predicted_noise
            self.fake_real_lr, self.predicted_kernel, self.predicted_noise = self.netG(self.H)
        else:
            self.E = self.netG(self.L)

        # if self.opt_train['task_name'] == 'een':
        #     # 先加上detach()做分离训练
        #     # 假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B. 那么可以这样做
        #     self.x_learned_lap, self.x_lap = self.netEEN(self.E.detach())
        #     self.finalSR = self.E.detach() + (self.x_learned_lap - self.x_lap)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        total_loss = 0
        E_loss = 0
        self.netG_forward()
        if current_step % 500 == 0:
            if self.opt_net['net_type'] == 'swinireen':
                self.tensorboard_logger.add_image('final_SR', tensor2img(self.x_finalSR.detach()[0].float().cpu()), 1,
                                                  dataformats='HWC')
                learned_lap = self.x_learned_lap.detach() - self.x_lap.detach()
                self.tensorboard_logger.add_image('learned_lap', tensor2img(learned_lap[0].float().cpu()), 1,
                                                  dataformats='HWC')
            else:
                self.tensorboard_logger.add_image('final_SR', tensor2img(self.E.detach()[0].float().cpu()), 1,
                                                  dataformats='HWC')
                self.tensorboard_logger.add_image('true_H', tensor2img(self.H.detach()[0].float().cpu()), 1,
                                                  dataformats='HWC')
                self.tensorboard_logger.add_image('true_L', tensor2img(self.L.detach()[0].float().cpu()), 1,
                                                  dataformats='HWC')

        # else:
        #     self.tensorboard_logger.add_image('final_SR', tensor2img(self.E.detach()[0].float().cpu()), 1,
        #                                   dataformats='HWC')
        # print('输出大小：', self.E.shape)
        # if self.opt_train['task_name'] == 'een':
        #     # 考虑利用 HR的边缘信息 与 强化后的边缘信息来计算 loss
        #     E_loss = 5 * self.EEN_lossfn(self.finalSR, self.H)
        #     if current_step % 500 == 0:
        #         self.tensorboard_logger.add_scalar('E_loss', torch.mean(E_loss.detach()), current_step)
        #     E_loss.backward()
        if self.opt_net['net_type'] == 'swinireen':

            G_loss = self.G_lossfn_weight * self.G_lossfn(self.x_base, self.H)
            E_loss = 5 * self.EEN_lossfn(self.x_finalSR, self.H)
            total_loss = G_loss + E_loss
            if current_step % 500 == 0:
                self.tensorboard_logger.add_scalar('G_loss', torch.mean(G_loss.detach()), current_step)
                self.tensorboard_logger.add_scalar('E_loss', torch.mean(E_loss.detach()), current_step)
                self.tensorboard_logger.add_scalar('total_loss', torch.mean(total_loss.detach()), current_step)
            total_loss.backward()

        elif self.opt_net['net_type'] == 'hybrid' or self.opt_net['net_type'] == 'hybridsr' or self.opt_net['net_type'] == 'swinunetir' or self.opt_net['net_type'] == 'allcnn' :
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            E_loss = 0
            # 添加边缘高频信息损失
            if self.opt_train['e_loss']:
                E_loss = self.E_lossfn_weight * self.E_lossfn(self.E, self.H)
            # 添加 感知损失
            if self.opt_train['p_loss']:
                P_loss = self.P_lossfn_weight * self.P_loss(self.E, self.H)
                total_loss = G_loss + E_loss + P_loss
            elif self.opt_train['e_loss']:
                total_loss = G_loss + E_loss
            else:
                total_loss = G_loss

            # 不同数据集同时跑时，关闭一个
            if current_step % 500 == 0:
                self.tensorboard_logger.add_scalar('G_loss', torch.mean(G_loss.detach()), current_step)
                # 添加边缘高频信息损失
                if self.opt_train['e_loss']:
                    self.tensorboard_logger.add_scalar('E_loss', torch.mean(E_loss.detach()), current_step)
                # 感知损失
                if self.opt_train['p_loss']:
                    self.tensorboard_logger.add_scalar('P_loss', torch.mean(P_loss.detach()), current_step)


                self.tensorboard_logger.add_scalar('total_loss', torch.mean(total_loss.detach()), current_step)

            # G_loss.backward()

            # 添加边缘高频信息损失
            total_loss.backward()

        elif self.opt_net['net_type'] == 'deg':
            G_loss = 0
            total_loss = 0
            l_d_total = 0
            for p in self.netD.parameters():
                p.requires_grad = False

            pred_g_fake = self.netD(self.fake_real_lr)
            G_loss = self.gan_loss(pred_g_fake, True)

            noise = self.predicted_noise
            noise_loss = self.nosie_loss(noise, torch.zeros_like(noise))

            total_loss = G_loss + 100*noise_loss
            if current_step % 500 == 0:
                self.tensorboard_logger.add_scalar('G_loss', torch.mean(G_loss.detach()), current_step)
                self.tensorboard_logger.add_image('final_SR', tensor2img(self.fake_real_lr.detach()[0].float().cpu()), 1,
                                                  dataformats='HWC')
            total_loss.backward()
            self.G_optimizer.step()

            # descriminator
            l_d_total = 0
            for p in self.netD.parameters():
                p.requires_grad = True

            self.D_optimizer.zero_grad()
            pred_d_real = self.netD(self.L)
            pred_d_fake = self.netD(self.fake_real_lr.detach())
            l_d_real = self.gan_loss(pred_d_real, True)
            l_d_fake = self.gan_loss(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
            if current_step % 500 == 0:
                self.tensorboard_logger.add_scalar('l_d_total', torch.mean(l_d_total.detach()), current_step)
            l_d_total.backward()
            self.D_optimizer.step()

        else:

            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            if current_step % 500 == 0:
                self.tensorboard_logger.add_scalar('G_loss', torch.mean(G_loss.detach()), current_step)
            G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)
        # if self.opt_train['task_name'] == 'een':
        #     self.E_optimizer.step()
        # if self.opt_net['net_type'] != 'deg':
        #     self.G_optimizer.step()
        self.G_optimizer.step()
        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        if self.opt_train['task_name'] == 'een':
            self.log_dict['E_loss'] = E_loss.item()
        else:
            self.log_dict['G_loss'] = G_loss.item()
            if self.opt_net['net_type'] == 'hybrid' or self.opt_net['net_type'] == 'hybridsr' or self.opt_net['net_type'] == 'allcnn':
                # 添加了边缘损失
                self.log_dict['total_loss'] = total_loss.item()
                if self.opt_train['e_loss']:
                    self.log_dict['E_loss'] = E_loss.item()
                if self.opt_train['p_loss']:
                    self.log_dict['P_loss'] = P_loss.item()


            elif self.opt_net['net_type'] == 'deg':
                self.log_dict['total_loss'] = total_loss.item()
                self.log_dict['D_loss'] = l_d_total.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        """
        # Kai Zhang, https://github.com/cszn/KAIR
        #
        # Args:
        #   init_type:
        #       default, none: pass init_weights
        #       normal; normal; xavier_normal; xavier_uniform;
        #       kaiming_normal; kaiming_uniform; orthogonal
        #   init_bn_type:
        #       uniform; constant
        #   gain:
        #       0.2
        """

        def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
            classname = m.__class__.__name__

            if classname.find('Conv') != -1 or classname.find('Linear') != -1:

                if init_type == 'normal':
                    init.normal_(m.weight.data, 0, 0.1)
                    m.weight.data.clamp_(-1, 1).mul_(gain)

                elif init_type == 'uniform':
                    init.uniform_(m.weight.data, -0.2, 0.2)
                    m.weight.data.mul_(gain)

                elif init_type == 'xavier_normal':
                    init.xavier_normal_(m.weight.data, gain=gain)
                    m.weight.data.clamp_(-1, 1)

                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=gain)

                elif init_type == 'kaiming_normal':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                    m.weight.data.clamp_(-1, 1).mul_(gain)

                elif init_type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                    m.weight.data.mul_(gain)

                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)

                else:
                    raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

                if m.bias is not None:
                    m.bias.data.zero_()

            elif classname.find('BatchNorm2d') != -1:

                if init_bn_type == 'uniform':  # preferred
                    if m.affine:
                        init.uniform_(m.weight.data, 0.1, 1.0)
                        init.constant_(m.bias.data, 0.0)
                elif init_bn_type == 'constant':
                    if m.affine:
                        init.constant_(m.weight.data, 1.0)
                        init.constant_(m.bias.data, 0.0)
                else:
                    raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

        if init_type not in ['default', 'none']:
            print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
            fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
            net.apply(fn)
        else:
            print('Pass this initialization! Initialization was done during network definition!')
