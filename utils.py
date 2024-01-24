import math
import os
import random
import shutil
from collections import OrderedDict

import torch
import scipy.signal
from torch import nn
from tqdm import tqdm
import cv2
import numpy as np
# import torch
# from medpy import metric
# from scipy.ndimage import zoom
# import torch.nn as nn
# import SimpleITK as sitk


# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes
#
#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()
#
#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss_ = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss_ = 1 - loss_
#         return loss_
#
#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss_ = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss_ += dice * weight[i]
#         return loss_ / self.n_classes
#
#
# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum() == 0:
#         return 1, 0
#     else:
#         return 0, 0
#
#
# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()#去掉bs
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)#相同sahpe，值全是0
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]#x=224,y=224
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0 ##输入改变形状匹配到224*224
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()#加上两个维度
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)#b,100,h,w
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()#b , h, w
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#
#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list
from skimage.metrics import mean_squared_error as mse_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk


def get_parent_dir(path=None, offset=-1):
    result = path if path else __file__
    for i in range(abs(offset)):
        result = os.path.dirname(result)
    return result


def reflesh_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def remove_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 80
    return 20 * np.log10(1 / np.math.sqrt(mse))


def psnr255(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 80
    return 20 * np.log10(1. / np.math.sqrt(mse))


def get_feature(image):
    # image = image[0:64,0:64]
    image = image.astype(np.float64)
    [m, n] = image.shape
    image = image.astype(np.float64)
    exp = sum(sum(image)) / (m * n)
    exp2 = sum(sum(image ** 2)) / (m * n)
    c2 = np.sqrt(np.sum((image - exp) ** 2))
    if c2 != 0:
        v2 = image - exp
        v2 = v2 / c2
    else:
        v2 = np.zeros([m, n])
    return exp, v2, c2


def fea2img(exp, v2, c2):
    img = exp + v2 * c2
    return img


def get_rand():
    di = random.random()
    if di <= 0.5:
        sign = -1.0
    else:
        sign = 1.0
    return sign


def get_LM(image):
    u = np.mean(np.mean(image))
    if u < 127:
        JND_LM = 17 * (1 - np.sqrt(u / 127)) + 3
    else:
        JND_LM = 3 * (u - 127) / 128 + 3
    return JND_LM


def get_sx(x_pre, exp_ori, c2_ori):
    x_pre = x_pre.astype(np.float64)
    exp_ori = exp_ori.astype(np.float64)
    c2_ori = c2_ori.astype(np.float64)
    return (x_pre - exp_ori) / c2_ori


def cut_images(folder, dst_folder, size, stride, mode, nums_of_img_dir, drop=False):
    """
    Batch crop images
    Args:
        folder: 文件夹
        dst_folder: 目标文件夹（保存分割好的图像）
        size: 保存的图像大小
        stride: 步长
        mode: train / val / test
        nums_of_img_dir: 图像个数
        drop: 是否丢弃一部分宽度

    Returns: 切割图片的个数

    """

    count = 0
    with tqdm(total=nums_of_img_dir) as pbar:
        for _, _, files in os.walk(folder):
            for file in files:
                image = cv2.imread(os.path.join(folder, file))
                if drop:
                    image = image[0:1024, :, :]  # [1024, 1080, 3]  [w, h, c]
                h_num = image.shape[0] // stride  # 1024 // 224 = 4  这个应该是w_num
                w_num = image.shape[1] // stride  # 1080 // 224 = 4  这个应该是h_num
                fore_name = os.path.splitext(file)[0]  # 文件名（不含扩展名）
                back_name = os.path.splitext(file)[-1]  # 扩展名
                for i in range(0, h_num):
                    if i * stride + size <= image.shape[0]:
                        for j in range(0, w_num):
                            if j * stride + size <= image.shape[1]:
                                tmp_img = image[i * stride:i * stride + size, j * stride:j * stride + size, :]
                                tmp_name = fore_name + '_' + str(i) + '_' + str(j) + back_name
                                if mode == "image":
                                    # folder = folder.replace('Ori', 'patches_image')
                                    cv2.imwrite(os.path.join(dst_folder, tmp_name), tmp_img)
                                    # folder = folder.replace('patches_image', 'Ori')
                                elif mode == "jnd":
                                    # folder = folder.replace('Dis', 'patches_jnd')
                                    cv2.imwrite(os.path.join(dst_folder, tmp_name), tmp_img)
                                    # folder = folder.replace('patches_jnd', 'Dis')
                                elif mode == "depth":
                                    # folder = folder.replace('Depth', 'patches_depth')
                                    cv2.imwrite(os.path.join(dst_folder, tmp_name), tmp_img)
                                    # folder = folder.replace('patches_depth', 'Depth')
                                elif mode == "seg":
                                    # folder = folder.replace('Seg', 'patches_seg')
                                    cv2.imwrite(os.path.join(dst_folder, tmp_name), tmp_img)
                                    # folder = folder.replace('patches_seg', 'Seg')
                                elif mode == "salient":
                                    # folder = folder.replace('Salient', 'patches_salient')
                                    cv2.imwrite(os.path.join(dst_folder, tmp_name), tmp_img)
                                    # folder = folder.replace('patches_salient', 'Salient')
                                count += 1
                pbar.set_description(
                    desc="%s is cut to save %s" % (os.path.join(folder, file), os.path.join(dst_folder, fore_name)))
                pbar.update(1)
    return count


def dnorm(x):
    x = (x + 1.) / 2.0
    return x.clamp_(0., 1.)


def normli(x):
    x = x * 2. - 1.
    return x.clamp_(-1., 1.)


def get_bg(img):
    mask = np.array([[1, 1, 1, 1, 1],
                     [1, 2, 2, 2, 1],
                     [1, 2, 0, 2, 1],
                     [1, 2, 2, 2, 1],
                     [1, 1, 1, 1, 1]])
    out = scipy.signal.convolve2d(img, mask, mode='same') / 32
    return out


def get_gm(img):
    G1 = np.array([[0, 0, 0, 0, 0],
                   [1, 3, 8, 3, 1],
                   [0, 0, 0, 0, 0],
                   [-1, -3, -8, -3, -1],
                   [0, 0, 0, 0, 0]])

    G2 = np.array([[0, 0, 1, 0, 0],
                   [0, 8, 3, 0, 0],
                   [1, 3, 0, -3, -1],
                   [0, 0, -3, -8, 0],
                   [0, 0, -1, 0, 0]])

    G3 = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 3, 8, 0],
                   [-1, -3, 0, 3, 1],
                   [0, -8, -3, 0, 0],
                   [0, 0, -1, 0, 0]])

    G4 = np.array([[0, 1, 0, -1, 0],
                   [0, 3, 0, -3, 0],
                   [0, 8, 0, -8, 0],
                   [0, 3, 0, -3, 0],
                   [0, 1, 0, -1, 0]])
    (H, W) = img.shape
    grad = np.zeros([H, W, 4])
    grad[:, :, 0] = scipy.signal.convolve2d(img, G1, mode='same') / 16
    grad[:, :, 1] = scipy.signal.convolve2d(img, G2, mode='same') / 16
    grad[:, :, 2] = scipy.signal.convolve2d(img, G3, mode='same') / 16
    grad[:, :, 3] = scipy.signal.convolve2d(img, G4, mode='same') / 16

    gm = np.max(abs(grad), axis=2)
    return gm


def get_JND(img):
    # 获取可见性阈值
    (H, W) = img.shape
    img_bg = get_bg(img)
    img_gm = get_gm(img)
    T0 = 17
    GAMMA = 4 / 128
    JNDl = GAMMA * (img_bg - 127) + 3
    JNDl[img_bg <= 127] = T0 * (1 - np.sqrt(img_bg[img_bg <= 127] / 127)) + 3

    LANDA = 1 / 2
    alpha = 0.0001 * img_bg + 0.115
    belta = LANDA - 0.01 * img_bg
    JNDt = img_gm * alpha + belta
    T = np.zeros([H, W, 2])

    T[:, :, 0] = JNDl
    T[:, :, 1] = JNDt
    C_TG = 0.3
    JND = np.sum(T, axis=2) - C_TG * np.min(T, axis=2)
    return JND


def get_JND_wusun(img):
    img = img.cpu()
    img = dnorm(img) * 255.  # 反标准化并反归一化

    (H, W) = img.shape
    img_bg = get_bg(img)
    img_gm = get_gm(img)
    T0 = 17
    GAMMA = 4 / 128
    JNDl = GAMMA * (img_bg - 127) + 3
    JNDl[img_bg <= 127] = T0 * (1 - np.sqrt(img_bg[img_bg <= 127] / 127)) + 3

    LANDA = 1 / 2
    alpha = 0.0001 * img_bg + 0.115
    belta = LANDA - 0.01 * img_bg
    JNDt = img_gm * alpha + belta
    T = np.zeros([H, W, 2])

    T[:, :, 0] = JNDl
    T[:, :, 1] = JNDt
    C_TG = 0.3
    JND = np.sum(T, axis=2) - C_TG * np.min(T, axis=2)
    Sign = np.random.random([H, W])
    Sign[Sign < 0.5] = -1
    # 注意这里
    Sign[Sign >= 0.5] = -1
    out = img + Sign * JND
    out = out / 255
    out = normli(out)
    out = out.cuda()
    return torch.clamp_(out, -1.0, 1.0)


def get_JND_patches(img):
    # 获取可见性阈值
    (H, W) = img.shape
    img_bg = get_bg(img)
    img_gm = get_gm(img)
    T0 = 17
    GAMMA = 4 / 128
    JNDl = GAMMA * (img_bg - 127) + 3
    JNDl[img_bg <= 127] = T0 * (1 - np.sqrt(img_bg[img_bg <= 127] / 127)) + 3

    LANDA = 1 / 2
    alpha = 0.0001 * img_bg + 0.115
    belta = LANDA - 0.01 * img_bg
    JNDt = img_gm * alpha + belta
    T = np.zeros([H, W, 2])

    T[:, :, 0] = JNDl
    T[:, :, 1] = JNDt
    C_TG = 0.3
    JND = np.sum(T, axis=2) - C_TG * np.min(T, axis=2)
    return JND


def get_JND_wusun_patches(img):
    (H, W) = img.shape
    img_bg = get_bg(img)
    img_gm = get_gm(img)
    T0 = 17
    GAMMA = 4 / 128
    JNDl = GAMMA * (img_bg - 127) + 3
    JNDl[img_bg <= 127] = T0 * (1 - np.sqrt(img_bg[img_bg <= 127] / 127)) + 3

    LANDA = 1 / 2
    alpha = 0.0001 * img_bg + 0.115
    belta = LANDA - 0.01 * img_bg
    JNDt = img_gm * alpha + belta
    T = np.zeros([H, W, 2])

    T[:, :, 0] = JNDl
    T[:, :, 1] = JNDt
    C_TG = 0.3
    JND = np.sum(T, axis=2) - C_TG * np.min(T, axis=2)
    Sign = np.random.random([H, W])
    Sign[Sign < 0.5] = -1
    # 这里注意
    Sign[Sign >= 0.5] = -1
    out = img + Sign * JND
    return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        model_dict = model.state_dict()
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        new_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(new_state_dict), len(new_dict)))
        # model.load_state_dict(new_dict)


def load_checkpoint_mlp_mixer(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint)
    except:
        model_dict = model.state_dict()
        state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        new_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(new_state_dict), len(new_dict)))
        # model.load_state_dict(new_dict)


def load_checkpoint_swimir(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint)
    except:
        model_dict = model.state_dict()
        state_dict = checkpoint["params"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        new_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(new_state_dict), len(new_dict)))
        # model.load_state_dict(new_dict)


def load_checkpoint_uformer_2(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint)
    except:
        model_dict = model.state_dict()
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        new_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(model_dict), len(new_dict)))
        # model.load_state_dict(new_dict)


def ssim_index_new(img1, img2, K, win):
    M, N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0] * 255) ** 2
    C2 = (K[1] * 255) ** 2
    win = win / np.sum(win)

    mu1 = scipy.signal.convolve2d(img1, win, mode='valid')
    mu2 = scipy.signal.convolve2d(img2, win, mode='valid')
    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1, img1), win, mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if (C1 > 0 and C2 > 0):
        ssim1 = 2 * sigma12 + C2
        ssim_map = np.divide(np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)),
                             np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2)))
        cs_map = np.divide((2 * sigma12 + C2), (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1, denominator2)
        # 如果index是真，就赋值，是假就原值
        n, m = mu1.shape
        for i in range(n):
            for j in range(m):
                if (index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j] * numerator2[i][j] / denominator1[i][j] * denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if ((denominator1[i][j] != 0) and (denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j] / denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if (denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j] / denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]

    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return mssim, mcs


# msssim.py


# ms_ssim实现一
def msssim(img1, img2):
    K = [0.01, 0.03]
    win = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    method = 'product'

    M, N = img1.shape
    H, W = win.shape

    downsample_filter = np.ones((2, 2)) / 4
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mssim_array = []
    mcs_array = []

    for i in range(0, level):
        mssim, mcs = ssim_index_new(img1, img2, K, win)
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2, ::2]
        img2 = filtered_im2[::2, ::2]

    # print(np.power(mcs_array[:level-1],weight[:level-1]))
    # print(mssim_array[level-1]**weight[level-1])
    overall_mssim = np.prod(np.power(mcs_array[:level - 1], weight[:level - 1])) * (
            mssim_array[level - 1] ** weight[level - 1])

    return overall_mssim


def switch_optimizer_tensor_to_device(optimizer):
    """
    将optimizer的tensor数据转到GPU上面去
    Args:
        optimizer:

    Returns:

    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def calculate_jnd_image(image1, image2):
    """
    计算两张图的JND图像
    Args:
        image1: 图1
        image2: 图2

    Returns: JND的灰度图

    """
    # 方法一
    image1_lab = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab).astype('float')
    image2_lab = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab).astype('float')

    diff = cv2.absdiff(image1_lab[:, :, 0], image2_lab[:, :, 0])
    diff_squared = cv2.pow(diff, 2.0)
    diff_sqrt = cv2.sqrt(diff_squared)
    normalized_diff = cv2.normalize(diff_sqrt, None, 0, 255, cv2.NORM_MINMAX)
    jnd_image = normalized_diff.astype('uint8')
    # 提高灰度图的整体亮度
    # jnd_image = np.clip(jnd_image + 50, 0, 255)

    # 方法二
    # image1_lab = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab).astype('float')
    # image2_lab = cv2.cvtColor(image2, cv2.COLOR_BGR2Lab).astype('float')
    # diff = cv2.absdiff(image1_lab[:, :, 0], image2_lab[:, :, 0])
    # normalized_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    # jnd_image2 = normalized_diff.astype('uint8')
    return jnd_image


def split_method_and_dataset(noise_img_path):
    """
    根据noise图像相对路径拆分出method和dataset
    Args:
        noise_img_path: noise图像相对路径

    Returns: method和dataset

    """
    prefix = True
    idx = 0
    method, dataset = '', ''
    for i in range(len(noise_img_path)):
        if noise_img_path[i] == '_':
            if prefix:
                prefix = False
                idx = i
            else:
                method = noise_img_path[idx + 1:i]
                dataset = noise_img_path[i + 1:].rstrip('/')
                break
    return method, dataset


def get_right_index(worksheet, row, max_column):
    """
    查找指定行最右边的数值所在列的索引
    Args:
        worksheet:
        row:
        max_column:

    Returns: 最右边的那个数值所在的列的索引

    """
    right_column = 1
    for column in range(max_column, 0, -1):
        cell_value = worksheet.cell(row=row, column=column).value
        if cell_value is not None:
            right_column = column
            break
    # print(max_column, right_column)
    return right_column


def get_worksheet_row_avg(worksheet, row, max_column):
    """
    根据某一行数据求平均值
    Args:
        worksheet:
        row:
        max_column:

    Returns:

    """
    total = 0.0
    for column in range(2, max_column):
        # print(type(worksheet.cell(row=row, column=column).value))
        cell_value = worksheet.cell(row=row, column=column).value
        if cell_value is not None:
            # print(cell_value, end=' ')
            # print("type(float(cell_value)) =", type(float(cell_value)))
            total += float(cell_value)

    return total / (max_column - 2)


def calculate_vdp(image1, image2):
    """
    calculate the value of Visual Difference Prediction
    Args:
        image1: [c,h,w]
        image2: [c,h,w]

    Returns:

    """
    # [c, h, w] => [h, w, c]
    image1 = image1.transpose(1, 2, 0)
    image2 = image2.transpose(1, 2, 0)

    # 将图像转换为灰度
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    optical_flow = cv2.calcOpticalFlowFarneback(gray_image1, gray_image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算光流的幅度
    flow_magnitude = np.sqrt(optical_flow[..., 0] ** 2 + optical_flow[..., 1] ** 2)

    # 计算视觉失真评分
    vdp_score = np.mean(flow_magnitude)

    return vdp_score


def find_the_second_char_for_string(s, c, order=2):
    t = order
    for i in range(len(s) - 1, -1, -1):
        if s[i] == c:
            t -= 1
            if t == 0:
                return i
    return -1
