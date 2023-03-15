import numpy as np
import os
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import utils
from DCBDNet import DCBDNet
from skimage import img_as_ubyte
import scipy.io as sio
from utils import utils_image as util

parser = argparse.ArgumentParser(description='Image Denoising using DCBDNet')
parser.add_argument('--input_dir', default='./Datasets/Set5/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Set5/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def noise():
    np.random.seed(seed=0)
    x = y = np.arange(start=-3, stop=3, step=0.0235)
    X, Y = np.meshgrid(x, y)
    Z = 3 * (1 - X) ** 2 * np.exp(-X ** 2 - (Y + 1) ** 2) - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(
        -X ** 2 - Y ** 2) - 1 / 3 * np.exp(-(X + 1) ** 2 - Y ** 2)
    P = 0 + (50 - 0) * ((Z - np.min(Z)) / (np.max(Z) - np.min(Z)))
    P1 = P / 255.
    Z1 = np.random.randn(256, 256)
    P2 = P1 * Z1

    noise = np.zeros((256, 256, 3))
    for i in range(3):
        noise[:, :, i] = P2
    return noise


def main():
    result_dir = os.path.join(args.result_dir)
    utils.mkdir(result_dir)

    model_restoration = DCBDNet(in_nc=3, out_nc=3, nc=64, bias=False)

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    H_path = os.path.join(args.input_dir, 'HR')
    L_path = os.path.join(args.input_dir, 'LR')
    E_path = os.path.join(args.result_dir)   # E_path, for Estimated images

    test_results = OrderedDict()
    test_results['psnr_noise'] = []
    test_results['psnr'] = []
    test_results['ssim_noise'] = []
    test_results['ssim'] = []

    H_paths = util.get_image_paths(H_path)

    for idx, img in enumerate(H_paths):

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)
        img_H = util.uint2single(img_H)

        img_L = img_H + noise()
        img_L = torch.from_numpy(img_L).unsqueeze(0).permute(0, 3, 1, 2).cuda().float()

        img_E, noise_level = model_restoration(img_L)

        img_E = torch.clamp(img_E, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)

        img_E = img_E.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        img_E = np.uint8((img_E * 255.0).round())

        img_H = util.imread_uint(H_paths[idx], n_channels=3)
        img_H = img_H.squeeze()

        img_L = torch.clamp(img_L, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        img_L = img_L.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        img_L = np.uint8((img_L * 255.0).round())

        psnr_noise = util.calculate_psnr(img_L, img_H, border=1)
        psnr = util.calculate_psnr(img_E, img_H, border=1)
        ssim_noise = util.calculate_ssim(img_L, img_H, border=1)
        ssim = util.calculate_ssim(img_E, img_H, border=1)
        test_results['psnr_noise'].append(psnr_noise)
        test_results['psnr'].append(psnr)
        test_results['ssim_noise'].append(ssim_noise)
        test_results['ssim'].append(ssim)
        print('{:s} - PSNR_Noise: {:.2f} dB; PSNR: {:.2f} dB; SSIM_Noise: {:.3f}; SSIM: {:.3f}.'.format(img_name + ext, psnr_noise, psnr, ssim_noise, ssim))

        util.imsave(img_L, os.path.join(L_path, img_name + ext))
        util.imsave(img_E, os.path.join(E_path, img_name + ext))

    ave_psnr_noise = sum(test_results['psnr_noise']) / len(test_results['psnr_noise'])
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim_noise = sum(test_results['ssim_noise']) / len(test_results['ssim_noise'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    print('Average PSNR/SSIM(RGB) - {} - PSNR_Noise: {:.2f} dB; PSNR: {:.2f} dB; SSIM_Noise: {:.4f}; SSIM: {:.4f}'.format('DCBDNet', ave_psnr_noise, ave_psnr, ave_ssim_noise, ave_ssim))


if __name__ == '__main__':
    main()




