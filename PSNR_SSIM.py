import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from sklearn import metrics


def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """

    n = len(y_true)
    mse = sum(np.square(y_true - y_pred)) / n
    return mse


def SAM_numpy(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始数据与重构数据的光谱角相似度，输出的数值的单位是角度制的°
    """
    M, N = x_true.shape[0], x_true.shape[1]

    prod_scal = np.sum(x_true * x_pred, axis=2)  # 星乘表示矩阵内各对应位置相乘, shape：[H,W]
    norm_orig, norm_fusa = np.sum(x_true * x_true, axis=2), np.sum(x_pred * x_pred, axis=2)
    prod_norm = np.sqrt(norm_orig * norm_fusa)

    prod_scal, prod_norm = prod_scal.reshape(M * N, 1), prod_norm.reshape(M * N, 1)

    z = np.where(prod_norm != 0)
    prod_norm, prod_scal = prod_norm[z], prod_scal[z]  # 把分母中出现0的地方剔除掉

    res = np.clip(prod_scal / prod_norm, -1, 1)
    res = np.arccos(res)  # 每一个位置的弧度
    sam = np.mean(res)  # 取平均
    sam = sam * 180 / math.pi  # 转换成角度°
    return sam


def SAM_torch(x_true, x_pred):

    dot_sum = torch.sum(x_true * x_pred, dim=1)
    norm_true = torch.norm(x_true, dim=1)
    norm_pred = torch.norm(x_pred, dim=1)

    res = dot_sum / (norm_pred * norm_true)
    res = torch.clamp(res,-1,1)
    res = torch.acos(res) * 180 / math.pi   # degree
    sam = torch.mean(res)
    return sam
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def calculate_metrics(img1, img2):
    psnr_value = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img1, img2)
    SAM = SAM_numpy(img1, img2)
    MSE = mse_value(img1, img2)
    return psnr_value, ssim_value, SAM,  MSE

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def batch_calculate_metrics(dir1, dir2, resize_to=(256, 256)):
    psnr_values = []
    ssim_values = []
    SAMs = []
    NRMSEs = []

    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))

    for file1, file2 in zip(files1, files2):
        path1 = os.path.join(dir1, file1)
        path2 = os.path.join(dir2, file2)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            print(f"Error reading images: {path1}, {path2}")
            continue

        img1 = resize_image(img1, resize_to)
        img2 = resize_image(img2, resize_to)

        psnr_value, ssim_value , SAM, NRMSE= calculate_metrics(img1, img2)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        SAMs.append(SAM)
        NRMSEs.append(NRMSE)


    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    mean_SAM = np.mean(SAMs)
    mean_NRMSE = np.mean(NRMSEs)

    return mean_psnr, mean_ssim,  mean_SAM, mean_NRMSE

# Example usage
dir1 = 'dataset_color/test/VH_color'
dir2 = 'results/VH'

mean_psnr, mean_ssim,SAM, NRMSE = batch_calculate_metrics(dir1, dir2, resize_to=(256, 256))
print(f"Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}, SAM: {SAM}, MSE:{NRMSE}")