# 测试
# 开发时间：2022/8/8 14:55
import os
import os.path as osp
from thop import profile
import glob
import sys
from collections import OrderedDict
from models.network_swinir import SwinIR as net
import cv2
import time
import datetime
import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models.network_hybrid import myModel
# from models.network_hybrid_depth_psnr import myModel as myModel_depth
from utilss import utils_image as util
from torchsummary import summary
from models.network_hat import HAT
from models.RRDB import GeneratorRRDB
from models.classical_SwinIR import SwinIR
from models.hnct import make_model
from models.nlsn import make_model as make_nlsn
from models.edsr import make_model as make_edsr
# from models.vdsr import make_model as make_vdsr
# from models.IMDTN import IMDTN as make_imdtn
from models.imdn_all import IMDN as make_imdn
from models.PAN import PAN as make_pan
# from models.CNN_SR import myNet
# from models.RFDN import make_model as make_rfdn
from models.BSRN import BSRN as make_bsrn
from swinUnetSR.HybridSR import mySRNet
# 不能在改进模型之后测试
model_path = '/home/yk/DiT-main/preTrainModels/HAT_x3/models/4215_G.pth'  # swinir_2322_G.pth HNCT_IR700_1554_G.pth IMDN_IR700_1806_G.pth models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#  /home/yk/Compare_models/YK-master/preTrainModels/PAN_x2-CVC09-800/models/1000_G.pth
# 基本网络结构已确定
# model_path = 'zoo_myModel/conv1-res-HBRT-128-ESA-edge-IR700-366_G.pth'

# X2
# model_path = 'zoo_model-X2/IMDN-x2-CVC09-1K_368_G.pth'

# 最近的尝试 X4
# model_path = 'zoo_newModel/IDHN-x4-ori-test-CVC-09-1K_672_G.pth'

# IDHN-0-5-BSConV-1008_G.pth  IDHN-0-5-BSRB-CVC-x2-752_G.pth  IDHN-noSpilt-SRB-Channel128-CVC-x2-965_G.pth#固定预训练模型
# model_path = 'shengao_newModel/ALL_CNN-emeb48-x4-992_G.pth'

device = torch.device('cuda:0')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')cuda:0

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
test_results['vif'] = []
test_results['psnr_y'] = []
test_results['ssim_y'] = []
test_results['psnr_b'] = []
psnr, vif, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0, 0


# ours
model = myModel()
## memory: 93644800 (result-C)

# model = mySRNet() ## LKFormer

# HAT
# model = HAT()

# RRDB
# model = GeneratorRRDB()

# classical swinIR
# model = SwinIR(upscale=2, img_size=64,
#                    window_size=8, img_range=1., depths=[6, 6, 6, 6,6,6],
#                    embed_dim=180, num_heads=[6, 6, 6, 6,6,6], mlp_ratio=2, upsampler='pixelshuffle')


# ALL-CNN
# FPS: 2.978578 avg time: 0.335731
# FPS: 3.016912 avg time: 0.331465 (使用了BSRB提取局部特征)
# FPS: 4.593392 avg time: 0.217704 (使用ESA替换PSAatten)
# FPS: 5.437396 avg time: 0.183912 (使用CCA替换PSAatten)
# FPS: 13.888752 avg time: 0.072001 (使用CCA替换PSAatten)
# FPS: 9.392680 avg time: 0.106466  (使用CCA替换PSAatten)
# FPS: 99.202814 avg time: 0.010080
# FPS: 72.235086 avg time: 0.013844 (CoorAtt替换CCA)
# FPS: 61.480062 avg time: 0.016265 (CCA)
# FPS: 106.354172 avg time: 0.009403 (embed_dim = 48)
# FPS: 111.773320 avg time: 0.008947 (embed_dim = 48)
# FPS: 114.247583 avg time: 0.008753  (embed_dim = 48)
# FPS: 56.850300 avg time: 0.017590 (LCB)
# FPS: 83.451601 avg time: 0.011983 (LCB)
# FPS: 90.450483 avg time: 0.011056 (LCB)
# FPS: 104.479692 avg time: 0.009571 (LCB base4)
# FPS: 64.010655 avg time: 0.015622 (LABP) memory:  322476544
# FPS: 116.935881 avg time: 0.008552  (LCB base=7)
# FPS: 61.406255 avg time: 0.016285 (LCB base=14)
# FPS: 123.224984 avg time: 0.008115 (ALLCNN_test20)
# model = myNet(inc=3,embed_dim=48)

# our_depth or our_channel128
# model = myModel_depth()

# HNCT
## memory: 156710400 (result-C)
# model = make_model()

# NLSN
# model = make_nlsn()

# EDSR
# model = make_edsr()

# IMDTN
# FPS: 7.229354 avg time: 0.138325
# model = make_imdtn()

# IMDN
# FPS: 2.965032 avg time: 0.337264
# FPS: 4.194119 avg time: 0.238429
# FPS: 8.598975 avg time: 0.116293
# FPS: 12.713793 avg time: 0.078655
# FPS: 108.759470 avg time: 0.009195
# FPS: 105.565368 avg time: 0.009473
# FPS: 89.282216 avg time: 0.011200
# FPS: 55.984686 avg time: 0.017862
# FPS: 95.387073 avg time: 0.010484
# FPS: 83.619031 avg time: 0.011959
## FPS: 88.544741 avg time: 0.011294 (used)
# memory:  102464000 (result-C)

# model = make_imdn()

# FPS: 74.088425 avg time: 0.013497
# FPS: 85.984893 avg time: 0.011630
# FPS: 81.160418 avg time: 0.012321
# FPS: 116.158649 avg time: 0.008609
# floaps:  1688984832.0
# params:  433448.0
# memory:  293592576
# model = make_rfdn()

# BSRN
# memory:  150255616 (result-C)
# floaps:  1365562368.0
# params:  352400.0
# model = make_bsrn()

# PAN
# FPS: 4.460569 avg time: 0.224187
# FPS: 3.331704 avg time: 0.300147
# FPS: 3.705777 avg time: 0.269849
# memory:  133305344 (result-C)
# FPS: 12.231772 avg time: 0.081754
# FPS: 62.653750 avg time: 0.015961
# model = make_pan()

# swinIR
## memory: 252309504 (result-C)
# model = net(upscale=4,
#             in_chans=3,
#             img_size=64,
#             window_size=8,
#             img_range=1.0,
#             depths=[6, 6, 6, 6],
#             embed_dim=60,
#             num_heads=[6, 6, 6, 6],
#             mlp_ratio=2,
#             upsampler="pixelshuffle",
#             resi_connection="1conv")

# 加载预训练模型
# param_key_g = 'params'
#
# pretrained_model = torch.load(model_path)
# model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
#                           strict=True)
#
# model.load_state_dict(torch.load(model_path), strict=True)

model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False

# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()

model = model.to(device)
# model = DataParallel(model)

# load_network(model_path, model, strict=True, param_key='params')
#
# model.eval()

print('Model path {:s}. \nTesting...'.format(model_path))

# 测试模型的大小
# input = torch.randn(1,3,64,64)
# input = input.to(device)
# floaps, params = profile(model,inputs=(input,))
# print('floaps: ', floaps)
# print('params: ', params)

# sys.exit(0)

idx = 0
# 计时开始
# time = util.timer()
# IR700
# for path in glob.glob(test_img_folder):
#     idx += 1
#     base = osp.splitext(osp.basename(path))[0]
#     # print('路径', path)
#     # print(idx, base)
#
#     # read images
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img * 1.0 / 255
#     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#     img_LR = img.unsqueeze(0)
#     img_LR = img_LR.to(device)
#     img_H = util.imread_uint(testHR + '/' + base + '.bmp', n_channels=3)
#     img_H = img_H.astype(np.float32)
#
#     with torch.no_grad():
#         # .data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         output = model(img_LR)
#     img_E = util.tensor2uint(output)
#     # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#     # output = (output * 255.0).round()
#     img_out = output
#     # -------------------------------
#     # 计算 PSNR SSIM VIF
#     psnr = util.calculate_psnr(img_E, img_H, border=4)
#     ssim = util.calculate_ssim(img_E, img_H, border=4)
#     vif = util.compare_vifp(img_E, img_H.astype(np.uint8))
#     test_results['psnr'].append(psnr)
#     test_results['ssim'].append(ssim)
#     test_results['vif'].append(vif)
#     print('Testing {:d}  - PSNR: {:.2f} dB; SSIM: {:.4f}; VIF: {:.5f};'
#           .
#           format(idx, psnr, ssim, vif))
#     # -------------------------------
#     cv2.imwrite('results/catCAPA_HBRT_xCA_edgeLoss_IR700/{:d}_HBRT.bmp'.format(idx), img_E)
#
# # summarize psnr/ssim
# if len(test_results['psnr']) is not None and len(test_results['ssim']) is not None:
#     ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#     ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#     ave_vif = sum(test_results['vif']) / len(test_results['vif'])
#     print(' \n-- Average PSNR/SSIM: {:.2f} dB; {:.4f};{:.5f}'.format(ave_psnr, ave_ssim, ave_vif))


# result-C / result-A X4
# res = []
# for img in util.get_image_paths(L_path):
#     # --------------------------------
#     # (1) img_L
#     # --------------------------------
#     idx += 1
#     img_name, ext = os.path.splitext(os.path.basename(img))
#     img_L = util.imread_uint(img, n_channels=3)
#     img_L = util.uint2tensor4(img_L)
#     img_L = img_L.to(device)
#     # DIV2K
#     # img_H = util.imread_uint(testHR + '/' + img_name[0:4]+ext, n_channels=3)
#
#     # result-C / result-A / IR700 unknow
#     img_H = util.imread_uint(testHR + '/' + img_name + ext, n_channels=3)
#
#     img_H = util.modcrop(img_H, 4)
#     img_H = img_H.astype(np.float32)
#     # --------------------------------
#     # (2) inference
#     # --------------------------------
#     torch.cuda.synchronize()
#     start = time.time()
#
#     img_E = model(img_L)
#
#     torch.cuda.synchronize()
#     end = time.time()
#     res.append(end - start)
#     print('memory: ', torch.cuda.max_memory_allocated())
#     # --------------------------------
#     # 计算 PSNR SSIM
#     img_out = img_E.squeeze(0).float().cpu().detach().numpy().transpose(1, 2, 0)
#     img_out = (img_out * 255.0).round()
#
#     psnr = util.calculate_psnr(img_out, img_H, border=4)
#     ssim = util.calculate_ssim(img_out, img_H, border=4)
#
#     test_results['psnr'].append(psnr)
#     test_results['ssim'].append(ssim)
#
#     # --------------------------------
#     # (3) img_E 计算 VIF
#     # --------------------------------
#     img_E = util.tensor2uint(img_E)
#     img_H = img_H.astype(np.uint8)
#     vif = util.compare_vifp(img_E, img_H)
#     test_results['vif'].append(vif)
#     # os.makedirs(E_path,exist_ok=True)
#     print('Testing {:s} {:d}  - PSNR: {:.2f} dB; SSIM: {:.4f};vif: {:.5f}; '
#           .
#           format(img_name, idx, psnr, ssim, vif))
#     # util.imsave(img_E, os.path.join(E_path, img_name  + '.png'))
#
#     # summarize psnr/ssim
# if len(test_results['psnr']) is not None and len(test_results['ssim']) is not None:
#     ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#     ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#     ave_vif = sum(test_results['vif']) / len(test_results['vif'])
#     print('\n{} \n-- Average PSNR/SSIM/VIF: {:.2f} dB; {:.4f}; {:.5f}'.format(L_path, ave_psnr, ave_ssim, ave_vif))


# IR100 以及其他测试集 X4
res = []
# E_path = '/home/yk/DiT-main/preTrainModels/HAT_x3/result-A_test'
# # results-C
# H_path = '/home/yk/Compare_models/YK-master/datasets/results-C'
# results-A
# H_path = '/home/yk/Compare_models/YK-master/datasets/results-A'

# 101
# H_path = '/home/yk/Compare_models/YK-master/datasets/101ThermalTau2'

## IR700
# H_path = '/home/yk/Compare_models/YK-master/datasets/IR700/train/valid_H'

# IR100
# H_path = '/home/yk/Compare_models/YK-master/datasets/IR100'
# E_path = '/home/yk/Compare_models/YK-master/results/IDHN-noSplit-SRB-channel128-CVC09-x2-IR100'

# DLS-NUC-100
# H_path = '/home/yk/Compare_models/YK-master/datasets/INFRARED100'
# E_path = '/home/yk/Compare_models/YK-master/results/IDHN-noSplit-SRB-channel128-CVC09-x2-DLS-NUC-100'

## CVC-09
H_path = '/home/yk/Compare_models/YK-master/datasets/CVC-09-1K/val'

sf = 4
border = sf
L_path = H_path
need_degradation = True
x8 = False
need_H = True
n_channels = 3
L_paths = util.get_image_paths(L_path)
H_paths = util.get_image_paths(H_path) if need_H else None
for idx, img in enumerate(L_paths):
    # --------------------------------
    # (1) img_L
    # --------------------------------
    img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
    img_L = util.imread_uint(img, n_channels=3)
    img_L = util.uint2single(img_L)

        # degradation process, bicubic downsampling
    if need_degradation:
            # print('img_L:',img_L.shape)
            img_L = util.modcrop(img_L, sf)
            # print('img_L1:', img_L.shape)
            img_L = util.imresize_np(img_L, 1 / sf)
            # print('img_L2:', img_L.shape)
            # img_L = util.uint2single(util.single2uint(img_L))
            # np.random.seed(seed=0)  # for reproducibility
            # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        # util.imshow(util.single2uint(img_L),
        #             title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)

    # --------------------------------
    # (2) inference
    # --------------------------------
    torch.cuda.synchronize()
    start = time.time()

    img_E = model(img_L)

    torch.cuda.synchronize()
    end = time.time()
    res.append(end - start)
    print('memory: ', torch.cuda.max_memory_allocated(device=1))
    # --------------------------------
    # img_H
    img_H = util.imread_uint(H_paths[idx], n_channels=3)
    img_H = img_H.squeeze()
    img_H = util.modcrop(img_H, sf)


    # 计算 PSNR SSIM
    img_out = img_E.squeeze(0).float().cpu().detach().numpy().transpose(1, 2, 0)
    img_out = (img_out * 255.0).round()

    psnr = util.calculate_psnr(img_out, img_H, border=4)
    ssim = util.calculate_ssim(img_out, img_H, border=4)

    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)

    # --------------------------------
    # (3) img_E 计算 VIF
    # --------------------------------
    img_E = util.tensor2uint(img_E)
    img_H = img_H.astype(np.uint8)
    vif = util.compare_vifp(img_E, img_H)
    test_results['vif'].append(vif)
    # os.makedirs(E_path,exist_ok=True)
    print('Testing {:s} {:d}  - PSNR: {:.2f} dB; SSIM: {:.4f};vif: {:.5f}; '
          .
          format(img_name, idx, psnr, ssim, vif))
    # util.imsave(img_E, os.path.join(E_path, img_name  + '.png'))

    # summarize psnr/ssim
if len(test_results['psnr']) is not None and len(test_results['ssim']) is not None:
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_vif = sum(test_results['vif']) / len(test_results['vif'])
    print('\n{} \n-- Average PSNR/SSIM/VIF: {:.2f} dB; {:.4f}; {:.5f}'.format(L_path, ave_psnr, ave_ssim, ave_vif))


# x2 上采样测试
# res = []
# result_name = 'IDHN-noSplit-SRB-Channel128-CVC-x2'
# sf = 2
# # results-C
# # H_path = '/home/yk/Compare_models/YK-master/datasets/results-C'  # os.path.join(testsets, testset_name) # L_path, for Low-quality images
# # E_path = '/home/yk/Compare_models/YK-master/results/IDHN-noSplit-SRB-Channel128-CVC-x2-results-C'
#
# # results-A
# H_path= '/home/yk/Compare_models/YK-master/datasets/results-A'
# E_path = '/home/yk/Compare_models/YK-master/results/IDHN-noSplit-SRB-Channel128-CVC-x2-results-A'
# border = sf
# L_path = H_path
# need_degradation = True
# x8 = False
# need_H = True
# n_channels = 3
# L_paths = util.get_image_paths(L_path)
# H_paths = util.get_image_paths(H_path) if need_H else None
#
# for idx, img in enumerate(L_paths):
#
#     # ------------------------------------
#     # (1) img_L
#     # ------------------------------------
#
#     img_name, ext = os.path.splitext(os.path.basename(img))
#     # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
#     img_L = util.imread_uint(img, n_channels=3)
#     img_L = util.uint2single(img_L)
#
#     # degradation process, bicubic downsampling
#     if need_degradation:
#         # print('img_L:',img_L.shape)
#         img_L = util.modcrop(img_L, sf)
#         # print('img_L1:', img_L.shape)
#         img_L = util.imresize_np(img_L, 1 / sf)
#         # print('img_L2:', img_L.shape)
#         # img_L = util.uint2single(util.single2uint(img_L))
#         # np.random.seed(seed=0)  # for reproducibility
#         # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
#
#     # util.imshow(util.single2uint(img_L),
#     #             title='LR image with noise level {}'.format(noise_level_img)) if show_img else None
#
#     img_L = util.single2tensor4(img_L)
#     img_L = img_L.to(device)
#
#     # ------------------------------------
#     # (2) img_E
#     # ------------------------------------
#
#     if not x8:
#         img_E = model(img_L)
#     else:
#         img_E = utils_model.test_mode(model, img_L, mode=3, sf=sf)
#
#     img_E = util.tensor2uint(img_E)
#
#     if need_H:
#
#         # --------------------------------
#         # (3) img_H
#         # --------------------------------
#
#         img_H = util.imread_uint(H_paths[idx], n_channels=3)
#         img_H = img_H.squeeze()
#         img_H = util.modcrop(img_H, sf)
#
#         # --------------------------------
#         # PSNR and SSIM
#         # --------------------------------
#         # print('SR:',img_E.shape)
#         # print('HR:', img_H.shape)
#         psnr = util.calculate_psnr(img_E, img_H, border=border)
#         ssim = util.calculate_ssim(img_E, img_H, border=border)
#         test_results['psnr'].append(psnr)
#         test_results['ssim'].append(ssim)
#         print('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name + ext, psnr, ssim))
#         # util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None
#
#         if np.ndim(img_H) == 3:  # RGB image
#             img_E_y = util.rgb2ycbcr(img_E, only_y=True)
#             img_H_y = util.rgb2ycbcr(img_H, only_y=True)
#             psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
#             ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=border)
#             test_results['psnr_y'].append(psnr_y)
#             test_results['ssim_y'].append(ssim_y)
#
#     # ------------------------------------
#     # save results
#     # ------------------------------------
#     os.makedirs(E_path, exist_ok=True)
#     util.imsave(img_E, os.path.join(E_path, img_name + '.png'))
#
# if need_H:
#     ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#     ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#     print(
#         'Average PSNR/SSIM(RGB) - {} - x{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr, ave_ssim))
#     if np.ndim(img_H) == 3:
#         ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
#         ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
#         print(
#             'Average PSNR/SSIM( Y ) - {} - x{} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr_y,
#                                                                                        ave_ssim_y))

# print(('Forward: {:.2f}s\n'.format(time.toc())))
time_sum = 0
for i in res:
    time_sum += i
print("FPS: %f"%(1.0/(time_sum/len(res))))
print("avg time: %f"%((time_sum/len(res))))
